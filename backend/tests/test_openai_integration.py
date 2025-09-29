"""
Integration tests for OpenAI-powered chatbot functionality
These tests verify that the OpenAI integration is working correctly with a real API key.
"""

import pytest
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from chatbot.openai_explainer import ARBharatChatbot


class TestOpenAIIntegration:
    """Integration tests for OpenAI functionality with real API"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.chatbot = ARBharatChatbot()
        self.sample_metadata = {
            'grid': {
                'type': 'square',
                'rows': 5,
                'cols': 5,
                'spacing': 40
            },
            'symmetry': {
                'rotational': 4,
                'reflection_axes': 2
            },
            'pattern': {
                'style': 'Traditional',
                'complexity': 'Medium'
            },
            'region_hint': 'Tamil Nadu'
        }
    
    @pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_openai_api_key_here',
        reason="OpenAI API key not configured"
    )
    def test_openai_client_initialization(self):
        """Test that OpenAI client initializes properly with API key"""
        assert self.chatbot.client is not None
        assert self.chatbot.model == 'gpt-4o-mini'  # Default model
    
    @pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_openai_api_key_here',
        reason="OpenAI API key not configured"
    )
    def test_explain_kolam_with_ai_real_response(self):
        """Test real AI-powered Kolam explanation"""
        if self.chatbot.client is None:
            pytest.skip("OpenAI client not available")
        
        result = self.chatbot.explain_kolam_with_ai(self.sample_metadata)
        
        # Basic checks for AI response
        assert isinstance(result, str)
        assert len(result) > 50  # Should be a substantial response
        assert "kolam" in result.lower() or "pattern" in result.lower()
        
    @pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_openai_api_key_here',
        reason="OpenAI API key not configured"
    )
    def test_chat_about_kolam_real_response(self):
        """Test real AI-powered conversational chat"""
        if self.chatbot.client is None:
            pytest.skip("OpenAI client not available")
        
        message = "What makes this pattern special?"
        result = self.chatbot.chat_about_kolam_with_ai(self.sample_metadata, message)
        
        # Basic checks for AI response
        assert isinstance(result, str)
        assert len(result) > 30  # Should be a meaningful response
        
    @pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_openai_api_key_here',
        reason="OpenAI API key not configured"
    )
    def test_generate_pattern_suggestions_real_response(self):
        """Test real AI-powered pattern suggestions"""
        if self.chatbot.client is None:
            pytest.skip("OpenAI client not available")
        
        user_preferences = "I like geometric patterns with cultural meaning"
        result = self.chatbot.generate_pattern_suggestions(user_preferences)
        
        # Basic checks for AI response
        assert isinstance(result, str)
        assert len(result) > 50  # Should be detailed suggestions
        
    @pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_openai_api_key_here',
        reason="OpenAI API key not configured"
    )
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        assert "AR BHARAT" in self.chatbot.system_prompt
        assert "Kolam" in self.chatbot.system_prompt
        assert "Rangoli" in self.chatbot.system_prompt
        assert "Indian" in self.chatbot.system_prompt
    
    def test_graceful_handling_without_api_key(self):
        """Test that system works gracefully without API key"""
        # Temporarily override API key
        original_key = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = 'invalid_key'
        
        try:
            # Create new chatbot instance
            test_chatbot = ARBharatChatbot()
            
            # Should still provide fallback responses
            result = test_chatbot.explain_kolam_with_ai(self.sample_metadata)
            assert isinstance(result, str)
            assert len(result) > 10  # Should have some content
            
        finally:
            # Restore original key
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
            else:
                os.environ.pop('OPENAI_API_KEY', None)


class TestOpenAIConnectionStatus:
    """Test the current status of OpenAI connection"""
    
    def test_api_key_status(self):
        """Check if API key is properly configured"""
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key or api_key == 'your_openai_api_key_here':
            pytest.skip("OpenAI API key not configured - this is expected if you haven't set it up yet")
        else:
            assert api_key.startswith('sk-'), "API key should start with 'sk-'"
            assert len(api_key) > 20, "API key should be substantial length"
            print(f"✅ OpenAI API key is configured (length: {len(api_key)})")
    
    def test_chatbot_initialization(self):
        """Test chatbot initialization regardless of API key status"""
        chatbot = ARBharatChatbot()
        
        # Should always initialize successfully
        assert chatbot is not None
        assert hasattr(chatbot, 'client')
        assert hasattr(chatbot, 'model')
        assert hasattr(chatbot, 'system_prompt')
        
        # Check client status
        if chatbot.client is not None:
            print("✅ OpenAI client initialized successfully")
        else:
            print("ℹ️  OpenAI client not available - using fallback mode")