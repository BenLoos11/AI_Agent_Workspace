import lmstudio as lms 
import traceback

print("Connecting to the model in LM Studio...")

try: 
    model = lms.llm()
    result = model.respond("Introduce yourself and explain that you are running locally.")

    print("\nModel response:")
    print(result)

except Exception as e: 
    print("\nCould not connect to the model")
    print("Check that a model is loaded and the LM Studio server is running")
    print(f"Details: {e}")
    traceback.print_exc()