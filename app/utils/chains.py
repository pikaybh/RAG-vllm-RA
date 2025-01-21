import types

__all__ = ["add_chain"]

# 메서드 추가
def add_chain(model, chain):
    """Adds a chain method to the model instance."""
    # logger.info(f"Adding chain: {chain.__name__}")
    bound_method = types.MethodType(chain, model)
    setattr(model, chain.__name__, bound_method)
    return model

if __name__ == "__main__":
    # 사용
    model = OpenAIModel(model_id="openai/gpt-4")
    add_chain(model, ra_chain)  # ra_chain을 model의 메서드로 추가