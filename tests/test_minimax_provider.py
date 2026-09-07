"""Unit and integration tests for MiniMax LLM provider in MiniOneRec."""

import json
import os
import re
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

# Mock heavy dependencies that may not be installed in test environment
_mock_modules = {}
for mod_name in [
    'torch', 'transformers', 'gensim', 'accelerate', 'accelerate.utils',
]:
    if mod_name not in sys.modules:
        _mock_modules[mod_name] = sys.modules[mod_name] = MagicMock()

# Add project root so we can import rq/text2emb/utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rq', 'text2emb'))

import utils as text2emb_utils


class TestGetResBatchDispatch(unittest.TestCase):
    """Test that get_res_batch correctly dispatches to the right provider."""

    @patch.object(text2emb_utils, 'get_openai_batch', return_value=["openai result"])
    def test_default_provider_is_openai(self, mock_openai):
        api_info = {"api_key_list": ["key"]}
        result = text2emb_utils.get_res_batch("model", ["prompt"], 100, api_info)
        mock_openai.assert_called_once()
        self.assertEqual(result, ["openai result"])

    @patch.object(text2emb_utils, 'get_deepseek_batch', return_value=["deepseek result"])
    def test_deepseek_provider(self, mock_ds):
        api_info = {"provider": "deepseek", "api_key_list": ["key"]}
        result = text2emb_utils.get_res_batch("model", ["prompt"], 100, api_info)
        mock_ds.assert_called_once()
        self.assertEqual(result, ["deepseek result"])

    @patch.object(text2emb_utils, 'get_minimax_batch', return_value=["minimax result"])
    def test_minimax_provider(self, mock_mm):
        api_info = {"provider": "minimax", "api_key_list": ["key"]}
        result = text2emb_utils.get_res_batch("model", ["prompt"], 100, api_info)
        mock_mm.assert_called_once()
        self.assertEqual(result, ["minimax result"])

    @patch.object(text2emb_utils, 'get_openai_batch', return_value=["fallback"])
    def test_unknown_provider_falls_back_to_openai(self, mock_openai):
        api_info = {"provider": "unknown_provider", "api_key_list": ["key"]}
        result = text2emb_utils.get_res_batch("model", ["prompt"], 100, api_info)
        mock_openai.assert_called_once()
        self.assertEqual(result, ["fallback"])


class TestMiniMaxBatch(unittest.TestCase):
    """Test get_minimax_batch function."""

    def _make_response(self, content, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return resp

    @patch.object(text2emb_utils.requests, 'post')
    def test_single_prompt_success(self, mock_post):
        mock_post.return_value = self._make_response("Hello world")
        api_info = {"provider": "minimax", "api_key_list": ["test-key"]}
        results = text2emb_utils.get_minimax_batch(
            "MiniMax-M2.7", ["test prompt"], 100, api_info
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "Hello world")

    @patch.object(text2emb_utils.requests, 'post')
    def test_multiple_prompts(self, mock_post):
        mock_post.side_effect = [
            self._make_response("Response 1"),
            self._make_response("Response 2"),
            self._make_response("Response 3"),
        ]
        api_info = {"provider": "minimax", "api_key_list": ["test-key"]}
        results = text2emb_utils.get_minimax_batch(
            "MiniMax-M2.7", ["p1", "p2", "p3"], 100, api_info
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "Response 1")
        self.assertEqual(results[1], "Response 2")
        self.assertEqual(results[2], "Response 3")

    @patch.object(text2emb_utils.requests, 'post')
    def test_custom_base_url(self, mock_post):
        mock_post.return_value = self._make_response("ok")
        api_info = {
            "provider": "minimax",
            "api_key_list": ["key"],
            "base_url": "https://custom.api.example.com/v1"
        }
        text2emb_utils.get_minimax_batch("MiniMax-M2.7", ["p"], 100, api_info)
        call_url = mock_post.call_args[0][0]
        self.assertTrue(call_url.startswith("https://custom.api.example.com/v1"))

    @patch.object(text2emb_utils.requests, 'post')
    def test_default_base_url(self, mock_post):
        mock_post.return_value = self._make_response("ok")
        api_info = {"provider": "minimax", "api_key_list": ["key"]}
        text2emb_utils.get_minimax_batch("MiniMax-M2.7", ["p"], 100, api_info)
        call_url = mock_post.call_args[0][0]
        self.assertEqual(call_url, "https://api.minimax.io/v1/chat/completions")

    @patch.object(text2emb_utils.requests, 'post')
    def test_empty_prompt_list(self, mock_post):
        api_info = {"provider": "minimax", "api_key_list": ["key"]}
        results = text2emb_utils.get_minimax_batch("MiniMax-M2.7", [], 100, api_info)
        self.assertEqual(results, [])
        mock_post.assert_not_called()


class TestSingleMiniMaxRequest(unittest.TestCase):
    """Test _single_minimax_request function."""

    def _make_response(self, content, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return resp

    @patch.object(text2emb_utils.requests, 'post')
    def test_successful_request(self, mock_post):
        mock_post.return_value = self._make_response("result text")
        api_info = {"api_key_list": ["test-key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "prompt", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "result text")

    @patch.object(text2emb_utils.requests, 'post')
    def test_think_tag_stripping(self, mock_post):
        content = "<think>internal reasoning here</think>The actual answer"
        mock_post.return_value = self._make_response(content)
        api_info = {"api_key_list": ["test-key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "prompt", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "The actual answer")

    @patch.object(text2emb_utils.requests, 'post')
    def test_multiline_think_tag_stripping(self, mock_post):
        content = "<think>\nStep 1: think\nStep 2: reason\n</think>\nFinal answer here"
        mock_post.return_value = self._make_response(content)
        api_info = {"api_key_list": ["test-key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "prompt", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "Final answer here")

    @patch.object(text2emb_utils.requests, 'post')
    def test_rate_limit_retry(self, mock_post):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        success = self._make_response("ok")
        mock_post.side_effect = [rate_limited, success]
        api_info = {"api_key_list": ["test-key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "prompt", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "ok")
        self.assertEqual(mock_post.call_count, 2)

    @patch.object(text2emb_utils.requests, 'post')
    def test_server_error_returns_empty(self, mock_post):
        error_resp = MagicMock()
        error_resp.status_code = 500
        mock_post.return_value = error_resp
        api_info = {"api_key_list": ["test-key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "prompt", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "")
        self.assertEqual(mock_post.call_count, 3)

    @patch.object(text2emb_utils.requests, 'post')
    def test_request_payload_format(self, mock_post):
        mock_post.return_value = self._make_response("ok")
        api_info = {"api_key_list": ["test-key"], "temperature": 0.5}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "test prompt", 256, api_info,
            "https://api.minimax.io/v1", 0
        )
        payload = mock_post.call_args[1]['json']
        self.assertEqual(payload['model'], "MiniMax-M2.7")
        self.assertEqual(payload['messages'][0]['content'], "test prompt")
        self.assertEqual(payload['max_tokens'], 256)
        self.assertEqual(payload['temperature'], 0.5)
        self.assertFalse(payload['stream'])

    @patch.object(text2emb_utils.requests, 'post')
    def test_authorization_header(self, mock_post):
        mock_post.return_value = self._make_response("ok")
        api_info = {"api_key_list": ["my-secret-key"]}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        headers = mock_post.call_args[1]['headers']
        self.assertEqual(headers['Authorization'], "Bearer my-secret-key")

    @patch.object(text2emb_utils.requests, 'post')
    def test_exception_retries(self, mock_post):
        mock_post.side_effect = [
            Exception("Connection error"),
            Exception("Timeout"),
            self._make_response("recovered")
        ]
        api_info = {"api_key_list": ["key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "recovered")

    @patch.object(text2emb_utils.requests, 'post')
    def test_no_think_tag_passthrough(self, mock_post):
        """Content without think tags should pass through unchanged."""
        mock_post.return_value = self._make_response("plain answer without thinking")
        api_info = {"api_key_list": ["key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "plain answer without thinking")

    @patch.object(text2emb_utils.requests, 'post')
    def test_whitespace_stripping(self, mock_post):
        mock_post.return_value = self._make_response("  answer with spaces  ")
        api_info = {"api_key_list": ["key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "answer with spaces")


class TestTemperatureClamping(unittest.TestCase):
    """Test MiniMax temperature clamping behavior."""

    @patch.object(text2emb_utils.requests, 'post')
    def test_temperature_clamped_to_max_1(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        api_info = {"api_key_list": ["key"], "temperature": 1.5}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        payload = mock_post.call_args[1]['json']
        self.assertLessEqual(payload['temperature'], 1.0)

    @patch.object(text2emb_utils.requests, 'post')
    def test_temperature_clamped_to_min_0(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        api_info = {"api_key_list": ["key"], "temperature": -0.5}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        payload = mock_post.call_args[1]['json']
        self.assertGreaterEqual(payload['temperature'], 0.0)

    @patch.object(text2emb_utils.requests, 'post')
    def test_default_temperature(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        api_info = {"api_key_list": ["key"]}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        payload = mock_post.call_args[1]['json']
        self.assertEqual(payload['temperature'], 0.4)

    @patch.object(text2emb_utils.requests, 'post')
    def test_temperature_0_accepted(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        api_info = {"api_key_list": ["key"], "temperature": 0.0}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        payload = mock_post.call_args[1]['json']
        self.assertEqual(payload['temperature'], 0.0)

    @patch.object(text2emb_utils.requests, 'post')
    def test_temperature_1_accepted(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        api_info = {"api_key_list": ["key"], "temperature": 1.0}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        payload = mock_post.call_args[1]['json']
        self.assertEqual(payload['temperature'], 1.0)


class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call real MiniMax API (skipped without API key)."""

    def setUp(self):
        self.api_key = os.environ.get("MINIMAX_API_KEY")
        if not self.api_key:
            self.skipTest("MINIMAX_API_KEY not set")

    def test_single_completion(self):
        api_info = {
            "provider": "minimax",
            "api_key_list": [self.api_key],
        }
        results = text2emb_utils.get_res_batch(
            "MiniMax-M3", ["Say 'hello' in one word."], 10, api_info
        )
        self.assertEqual(len(results), 1)
        self.assertIn("hello", results[0].lower())

    def test_batch_completion(self):
        api_info = {
            "provider": "minimax",
            "api_key_list": [self.api_key],
        }
        prompts = [
            "What is 1+1? Answer with just the number.",
            "What is 2+2? Answer with just the number.",
        ]
        results = text2emb_utils.get_res_batch(
            "MiniMax-M3", prompts, 64, api_info
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(len(results[0]) > 0)
        self.assertTrue(len(results[1]) > 0)

    def test_end_to_end_preference_prompt(self):
        """Test with a prompt similar to actual MiniOneRec usage."""
        api_info = {
            "provider": "minimax",
            "api_key_list": [self.api_key],
        }
        prompt = text2emb_utils.intention_prompt.format(
            dataset_full_name="Electronics",
            item_title="Wireless Bluetooth Headphones",
            review="Great sound quality and comfortable fit. Battery lasts all day."
        )
        results = text2emb_utils.get_res_batch(
            "MiniMax-M3", [prompt], 256, api_info
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(len(results[0]) > 0)
        # The response should contain preference analysis
        result_lower = results[0].lower()
        self.assertTrue(
            "preference" in result_lower or "characteristics" in result_lower or "item" in result_lower
        )


class TestTestGenerationConfigHasNoTopKTopP(unittest.TestCase):
    """Ensure test_generation_config disables top_k/top_p to avoid no-valid-token warnings."""

    def test_trainer_source_sets_top_k_and_top_p_none(self):
        """Regression test for issue #66: test_generation_config must set top_k=None, top_p=None."""
        import ast, os
        src = os.path.join(os.path.dirname(__file__), '..', 'minionerec_trainer.py')
        with open(src) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == 'test_generation_config':
                        call = node.value
                        if isinstance(call, ast.Call):
                            kw_map = {kw.arg: kw.value for kw in call.keywords}
                            self.assertIn('top_k', kw_map, "test_generation_config must include top_k=None")
                            self.assertIn('top_p', kw_map, "test_generation_config must include top_p=None")
                            self.assertIsInstance(kw_map['top_k'], ast.Constant)
                            self.assertIsNone(kw_map['top_k'].value, "top_k must be None")
                            self.assertIsInstance(kw_map['top_p'], ast.Constant)
                            self.assertIsNone(kw_map['top_p'].value, "top_p must be None")
                            return
        self.fail("Could not find test_generation_config assignment in minionerec_trainer.py")


if __name__ == '__main__':
    unittest.main()
