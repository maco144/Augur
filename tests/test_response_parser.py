"""Tests for the Augur specialist response parser."""
import json

from augur.response_parser import parse_specialist_response


class TestCleanJson:

    def test_valid_json_parsed_successfully(self):
        raw = json.dumps({
            "probability": 0.75,
            "confidence": 0.8,
            "reasoning": "Strong historical trend.",
            "key_assumptions": ["economy stays stable"],
            "key_uncertainties": ["election outcome"],
            "would_change_if": "major policy shift",
        })
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert abs(result["probability"] - 0.75) < 0.001
        assert abs(result["confidence"] - 0.8) < 0.001
        assert result["reasoning"] == "Strong historical trend."
        assert result["key_assumptions"] == ["economy stays stable"]
        assert result["key_uncertainties"] == ["election outcome"]
        assert result["would_change_if"] == "major policy shift"


class TestMarkdownFenced:

    def test_json_language_tag(self):
        raw = '```json\n{"probability": 0.6, "confidence": 0.7, "reasoning": "test"}\n```'
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert abs(result["probability"] - 0.6) < 0.001

    def test_no_language_tag(self):
        raw = '```\n{"probability": 0.55, "confidence": 0.65, "reasoning": "no tag"}\n```'
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert abs(result["probability"] - 0.55) < 0.001


class TestXmlWrapped:

    def test_json_tag(self):
        raw = '<json>{"probability": 0.4, "confidence": 0.5, "reasoning": "xml test"}</json>'
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert abs(result["probability"] - 0.4) < 0.001

    def test_response_tag(self):
        raw = '<response>\n{"probability": 0.3, "confidence": 0.9, "reasoning": "resp tag"}\n</response>'
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert abs(result["probability"] - 0.3) < 0.001


class TestSurroundingText:

    def test_json_with_explanation(self):
        raw = (
            'Here is my analysis:\n'
            '{"probability": 0.65, "confidence": 0.7, "reasoning": "surrounded"}\n'
            'I hope this helps!'
        )
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert abs(result["probability"] - 0.65) < 0.001
        assert result["reasoning"] == "surrounded"


class TestCoercion:

    def test_string_probability_coerced(self):
        raw = json.dumps({"probability": "0.7", "confidence": "0.8", "reasoning": "strings"})
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert abs(result["probability"] - 0.7) < 0.001
        assert abs(result["confidence"] - 0.8) < 0.001

    def test_probability_clamped_high(self):
        raw = json.dumps({"probability": 1.5, "confidence": 0.5, "reasoning": "too high"})
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert result["probability"] == 1.0

    def test_probability_clamped_low(self):
        raw = json.dumps({"probability": -0.3, "confidence": 0.5, "reasoning": "too low"})
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert result["probability"] == 0.0

    def test_confidence_clamped_high(self):
        raw = json.dumps({"probability": 0.5, "confidence": 2.0, "reasoning": "over"})
        result = parse_specialist_response(raw)
        assert result["confidence"] == 1.0

    def test_confidence_clamped_low(self):
        raw = json.dumps({"probability": 0.5, "confidence": -1.0, "reasoning": "under"})
        result = parse_specialist_response(raw)
        assert result["confidence"] == 0.0


class TestMissingFields:

    def test_missing_fields_get_defaults(self):
        raw = json.dumps({"probability": 0.6})
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert abs(result["probability"] - 0.6) < 0.001
        assert result["confidence"] == 0.5  # default
        assert result["reasoning"] == ""
        assert result["key_assumptions"] == []
        assert result["key_uncertainties"] == []
        assert result["would_change_if"] is None

    def test_empty_object_gets_all_defaults(self):
        raw = "{}"
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert result["probability"] == 0.5
        assert result["confidence"] == 0.5


class TestParseError:

    def test_garbage_input(self):
        raw = "This is just random text with no JSON at all."
        result = parse_specialist_response(raw)
        assert result["status"] == "parse_error"
        assert result["probability"] == 0.5
        assert result["confidence"] == 0.0
        assert result["reasoning"] == "Unparseable response."

    def test_empty_string(self):
        result = parse_specialist_response("")
        assert result["status"] == "parse_error"

    def test_whitespace_only(self):
        result = parse_specialist_response("   \n\t  ")
        assert result["status"] == "parse_error"


class TestNestedJson:

    def test_nested_json_extracts_top_level(self):
        inner = json.dumps({"nested": True})
        raw = json.dumps({
            "probability": 0.8,
            "confidence": 0.9,
            "reasoning": f"Analysis based on {inner}",
            "key_assumptions": ["stable conditions"],
            "key_uncertainties": [],
            "would_change_if": None,
        })
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert abs(result["probability"] - 0.8) < 0.001


class TestMultipleJsonBlocks:

    def test_picks_block_with_probability(self):
        raw = (
            'Some preamble.\n'
            '{"name": "metadata", "version": 1}\n'
            'More text.\n'
            '{"probability": 0.72, "confidence": 0.85, "reasoning": "correct block"}\n'
            'Trailing text.'
        )
        result = parse_specialist_response(raw)
        assert result["status"] == "success"
        assert abs(result["probability"] - 0.72) < 0.001
        assert result["reasoning"] == "correct block"
