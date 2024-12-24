import pytest
from src.data_preparation import format_alpaca_prompt, formatting_prompts_func
from transformers import AutoTokenizer

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

def test_format_alpaca_prompt():
    instruction = "Test instruction"
    input_text = "Test input"
    output = "Test output"
    
    formatted = format_alpaca_prompt(instruction, input_text, output)
    
    assert "### Instruction:" in formatted
    assert "### Input:" in formatted
    assert "### Response:" in formatted
    assert instruction in formatted
    assert input_text in formatted
    assert output in formatted

def test_formatting_prompts_func(tokenizer):
    examples = {
        "en_instruction": ["Test EN instruction"],
        "en_input": ["Test EN input"],
        "en_output": ["Test EN output"],
        "zh_instruction": ["测试中文指令"],
        "zh_input": ["测试中文输入"],
        "zh_output": ["测试中文输出"]
    }
    
    result = formatting_prompts_func(examples, tokenizer)
    
    assert "text" in result
    assert len(result["text"]) == 2  # One English and one Chinese example
    assert isinstance(result["text"], list)