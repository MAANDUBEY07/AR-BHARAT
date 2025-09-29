#!/usr/bin/env python3
"""
Debug the chatbot issue directly in the backend environment
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from chatbot.openai_explainer import ar_bharat_chatbot

def test_direct_call():
    """Test the chatbot function directly"""
    try:
        print("Testing direct AR BHARAT chatbot call...")
        print("="*50)
        
        # Test with empty metadata (same as frontend)
        metadata = {}
        message = "How do I create better patterns?"
        history = []
        
        print(f"Message: {message}")
        print(f"Metadata: {metadata}")
        print(f"Use AI: True")
        print("\n" + "-"*30)
        
        # Call the function directly
        result = ar_bharat_chatbot.chat_about_kolam_with_ai(metadata, message, history)
        
        print("Response:")
        print(result)
        print("\n" + "="*50)
        
        # Check if fallback was triggered
        if "(AI chat temporarily unavailable)" in result:
            print("🚨 FALLBACK TRIGGERED - OpenAI failed!")
            
            # Let's try to diagnose why
            print("\nDiagnosing OpenAI client...")
            print(f"Client exists: {ar_bharat_chatbot.client is not None}")
            print(f"Model: {ar_bharat_chatbot.model}")
            
            if ar_bharat_chatbot.client is not None:
                print("Client is available, but API call might have failed")
            else:
                print("Client initialization failed")
        else:
            print("✅ OpenAI working correctly!")
            
    except Exception as e:
        print(f"❌ Exception in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_call()