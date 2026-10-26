import sys
import os
sys.path.append(os.path.dirname(__file__))

from model import TransformerEvaluatorService

def main():
    print("--- Testing TransformerEvaluatorService ---")
    service = TransformerEvaluatorService("distilgpt2")
    
    test_text = "The cat is sleeping on the mat because it is tired."
    print(f"\nEvaluating Input: '{test_text}'")
    res = service.evaluate_text(test_text, threshold=0.50)
    
    print("\n--- RESULTS ---")
    print(f"Original Tokens: {res['original_tokens']}")
    print(f"Selected Tokens: {res['selected_tokens']}")
    print(f"Tokens Saved: {res['tokens_saved']}")
    print(f"Saving Percentage: {res['saving_percentage']}%")
    print(f"Retained Tokens: {res['retained_tokens']}")
    print(f"Filtered Tokens: {res['filtered_tokens']}")
    print(f"Chatbot Response: {res['chatbot_response']}")
    print("\nToken Table:")
    for item in res['token_analysis']:
        print(f"  {item['token']:<12} | Score: {item['confidence']:.2f} | Status: {item['status']}")

    assert res['tokens_saved'] == res['original_tokens'] - res['selected_tokens'], "Math mismatch!"
    print("\n[SUCCESS] ALL BACKEND TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
