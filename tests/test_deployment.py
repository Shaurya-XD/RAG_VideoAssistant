"""Offline checks: real Chroma storage, fake embeddings/models, no API credits."""

import asyncio
import gc
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

from core.rag_engine import ask_question, build_rag_chain
from core.settings import MAX_QUESTION_CHARS, MAX_TRANSCRIPT_CHARS
from core.transcriber import transcribe_video
from core.vector_store import build_vector_store

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.embedding_patch = patch(
            "core.vector_store.get_embeddings",
            return_value=DeterministicFakeEmbedding(size=16),
        )
        self.embedding_patch.start()
        self.addCleanup(self.embedding_patch.stop)

    def test_concurrent_videos_stay_separate_and_cleanup_is_isolated(self):
        texts = ["Video A: Saturn is a planet.", "Video B: whales live in oceans."]
        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = list(executor.map(build_vector_store, texts))
        self.assertNotEqual(first._collection.name, second._collection.name)
        # Even a question about the other video cannot retrieve its documents.
        self.assertEqual(first.similarity_search("whales", k=1)[0].page_content, texts[0])
        self.assertEqual(second.similarity_search("Saturn", k=1)[0].page_content, texts[1])
        client, first_name = first._client, first._collection.name
        del first
        gc.collect()
        self.assertNotIn(first_name, [c.name for c in client.list_collections()])
        self.assertEqual(second.get()["documents"], [texts[1]])

    def test_failed_embedding_build_does_not_remove_another_video(self):
        store = build_vector_store("Keep this existing workspace.")
        before = {c.name for c in store._client.list_collections()}
        embeddings = Mock()
        embeddings.embed_documents.side_effect = RuntimeError("Provider unavailable")
        with patch("core.vector_store.get_embeddings", return_value=embeddings):
            with self.assertRaises(RuntimeError):
                build_vector_store("This new workspace fails to index.")
        self.assertEqual(before, {c.name for c in store._client.list_collections()})
        self.assertEqual(store.get()["documents"], ["Keep this existing workspace."])

    def test_chain_retains_its_index_until_released(self):
        # Echo the formatted prompt so this test can inspect retrieved context.
        llm = RunnableLambda(lambda prompt: AIMessage(
            content=prompt.to_string(), response_metadata={"finish_reason": "stop"}
        ))
        with patch("core.rag_engine.get_llm", return_value=llm):
            first = build_rag_chain("First video's private source.")
            second = build_rag_chain("Second video's different source.")
        gc.collect()
        result = ask_question(first, "What was discussed?")
        self.assertIn("First video's private source.", result)
        self.assertNotIn("Second video's different source.", result)
        del second
        gc.collect()
        self.assertIn("First video's private source.", ask_question(first, "Explain."))

    def test_input_limits_reject_before_api_calls(self):
        for text in (" ", "x" * (MAX_TRANSCRIPT_CHARS + 1)):
            with self.assertRaises(ValueError):
                build_vector_store(text)
        chain = Mock()
        with self.assertRaises(ValueError):
            ask_question(chain, "x" * (MAX_QUESTION_CHARS + 1))
        chain.invoke.assert_not_called()


class TranscriptionTests(unittest.TestCase):
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake", "GEMINI_MODEL": "fake"})
    @patch("core.transcriber.ChatGoogleGenerativeAI")
    def test_incomplete_response_is_never_accepted(self, model):
        model.return_value.invoke.return_value = AIMessage(
            content="Partial text", response_metadata={"finish_reason": "MAX_TOKENS"}
        )
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            transcribe_video("https://www.youtube.com/watch?v=IPJdfGtFsuI")
        model.return_value.invoke.return_value = AIMessage(
            content="Complete text", response_metadata={"finish_reason": "STOP"}
        )
        self.assertEqual(
            transcribe_video("https://www.youtube.com/watch?v=IPJdfGtFsuI"), "Complete text"
        )


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        # AppTest doesn't need a network event loop. Avoid sandbox socket issues.
        loop = Mock(spec=asyncio.AbstractEventLoop)
        loop.is_closed.return_value = False
        loop.is_running.return_value = False
        loop_patch = patch("asyncio.new_event_loop", return_value=loop)
        loop_patch.start()
        self.addCleanup(loop_patch.stop)
        env_patch = patch.dict(os.environ, {
            "GEMINI_API_KEY": "fake", "GEMINI_MODEL": "fake",
            "GEMINI_EMBEDDING_MODEL": "fake", "GROQ_API_KEY": "fake", "GROQ_MODEL": "fake",
        })
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def click(self, app, label):
        next(button for button in app.button if button.label == label).click().run()
        self.assertFalse(app.exception)

    def test_ui_flow_reuses_chain_and_reset_discards_it(self):
        # Exercise the real key-points function and prompt, mocking only the LLM.
        # A missing UI import must fail this check rather than hiding behind
        # the UI's generic error handler.
        points = Mock(return_value=AIMessage(
            content="- Main point", response_metadata={"finish_reason": "stop"}
        ))
        with patch("core.transcriber.transcribe_video", return_value="A video transcript."), \
             patch("core.rag_engine.build_rag_chain", return_value=Mock()) as build, \
             patch("core.rag_engine.ask_question", return_value="A supported answer."), \
             patch("core.summarise.summarise_transcript", return_value="A concise summary."), \
             patch("core.rag_engine.get_llm", return_value=RunnableLambda(points)):
            app = AppTest.from_file(APP_PATH, default_timeout=30).run()
            self.assertFalse(app.exception)
            app.text_input[0].set_value("https://youtu.be/IPJdfGtFsuI")
            self.click(app, "Analyze video →")
            self.click(app, "Generate summary")
            self.click(app, "Extract key points")
            language = "the same language as the source"
            self.assertFalse(app.error)
            self.assertEqual(app.session_state["workspace"]["key_points"][language], "- Main point")
            points.assert_called_once()
            self.assertIn("A concise summary.", points.call_args.args[0].to_string())
            app.run()
            self.assertTrue(next(b for b in app.button if b.label == "Extract key points").disabled)
            points.assert_called_once()
            self.click(app, "Prepare video questions")
            app.chat_input[0].set_value("Explain the main point.").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state["workspace"]["messages"]), 2)
            app.run()
            build.assert_called_once_with("A video transcript.")
            self.click(app, "＋ New video")
            self.assertIsNone(app.session_state["workspace"]["rag_chain"])
            self.assertEqual(app.session_state["workspace"]["transcript"], "")

    def test_key_points_api_failure_preserves_summary_and_allows_retry(self):
        points = Mock(side_effect=[
            RuntimeError("429 rate limit"),
            AIMessage(content="- Recovered point", response_metadata={"finish_reason": "stop"}),
        ])
        with patch("core.rag_engine.get_llm", return_value=RunnableLambda(points)):
            app = AppTest.from_file(APP_PATH, default_timeout=30).run()
            language = "English"
            workspace = app.session_state["workspace"]
            workspace.update(url="https://www.youtube.com/watch?v=IPJdfGtFsuI", transcript="Source text.")
            workspace["summaries"][language] = "Keep this summary."
            app.run()
            app.selectbox[0].set_value(language).run()
            self.click(app, "Extract key points")
            self.assertTrue(any("usage limit" in error.value for error in app.error))
            self.assertNotIn(language, workspace["key_points"])
            self.assertEqual(workspace["summaries"][language], "Keep this summary.")
            self.assertFalse(next(b for b in app.button if b.label == "Extract key points").disabled)
            self.click(app, "Extract key points")
            self.assertFalse(app.error)
            self.assertEqual(app.session_state["workspace"]["key_points"][language], "- Recovered point")

    def test_workspace_opens_directly_without_sidebar(self):
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertIn("workspace", app.session_state)
        self.assertEqual(len(app.sidebar), 0)
        self.assertEqual([field.label for field in app.text_input], ["YouTube video URL"])
        self.assertEqual([title.value for title in app.title], ["Video Notes"])


if __name__ == "__main__":
    unittest.main()
