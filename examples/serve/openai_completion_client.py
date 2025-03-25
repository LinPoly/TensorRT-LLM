from openai import OpenAI


def run_completion(max_tokens: int):
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="tensorrt_llm",
    )
    response = client.completions.create(
        model="TinyLlama-1.1B-Chat-v1.0",
        prompt="Where is New York?",
        max_tokens=max_tokens,
    )
    return response


if __name__ == "__main__":
    print(run_completion(max_tokens=20))
