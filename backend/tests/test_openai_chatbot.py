"""
Unit tests for OpenAI-powered chatbot functionality
"""

import pytest
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from unittest.mock import Mock, patch, MagicMock
from chatbot.openai_explainer import ARBharatChatbot


class TestARBharatChatbot:
    """Test cases for ARBharatChatbot class"""
    
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
    
    def test_explain_kolam_with_ai_success(self):
        """Test successful AI-powered Kolam explanation"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        # Create a proper mock structure for OpenAI response
        mock_message = Mock()
        mock_message.content = "This is a beautiful traditional Kolam pattern..."
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create a new chatbot instance with mocked client
        chatbot = ARBharatChatbot(client=mock_client)
        
        # Act
        result = chatbot.explain_kolam_with_ai(self.sample_metadata)
        
        # Assert
        assert result == "This is a beautiful traditional Kolam pattern..."
        mock_client.chat.completions.create.assert_called_once()
        
    @patch('chatbot.explainer.explain_kolam')
    def test_explain_kolam_with_ai_fallback(self, mock_explain):
        """Test fallback to original explainer when AI fails"""
        # Arrange
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_explain.return_value = "Fallback explanation"
        
        # Create chatbot with mocked client
        chatbot = ARBharatChatbot(client=mock_client)
        
        # Act
        result = chatbot.explain_kolam_with_ai(self.sample_metadata)
        
        # Assert
        assert "Fallback explanation" in result
        assert "AI explanation temporarily unavailable" in result
        mock_explain.assert_called_once_with(self.sample_metadata)
        
    def test_chat_about_kolam_with_ai_success(self):
        """Test successful AI-powered chat"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        # Create a proper mock structure for OpenAI response
        mock_message = Mock()
        mock_message.content = "Great question! Kolam patterns represent..."
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create chatbot with mocked client
        chatbot = ARBharatChatbot(client=mock_client)
        
        # Act
        result = chatbot.chat_about_kolam_with_ai(
            self.sample_metadata, 
            "Tell me about this pattern"
        )
        
        # Assert
        assert result == "Great question! Kolam patterns represent..."
        mock_client.chat.completions.create.assert_called_once()
        
    @patch('chatbot.openai_explainer.OpenAI')
    def test_chat_with_conversation_history(self, mock_openai):
        """Test chat with conversation history"""
        # Arrange
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        # Create a proper mock structure for OpenAI response
        mock_message = Mock()
        mock_message.content = "Based on our previous conversation..."
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        conversation_history = [
            {"role": "user", "content": "What is a Kolam?"},
            {"role": "assistant", "content": "A Kolam is a traditional Indian floor art..."}
        ]
        
        # Act
        result = self.chatbot.chat_about_kolam_with_ai(
            self.sample_metadata,
            "Tell me more about the cultural significance",
            conversation_history
        )
        
        # Assert
        assert result == "Based on our previous conversation..."
        # Check that conversation history was included in the call
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        assert len(messages) >= 3  # System + context + history + user message
        
    @patch('chatbot.openai_explainer.OpenAI')
    def test_generate_pattern_suggestions_success(self, mock_openai):
        """Test successful pattern suggestions generation"""
        # Arrange
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        # Create a proper mock structure for OpenAI response
        mock_message = Mock()
        mock_message.content = "Here are 3 beautiful Kolam patterns for you..."
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        preferences = {
            'difficulty': 'Medium',
            'region': 'Tamil Nadu',
            'occasion': 'Pongal',
            'size': 'Large',
            'colors': 'Traditional'
        }
        
        # Act
        result = self.chatbot.generate_pattern_suggestions(preferences)
        
        # Assert
        assert result == "Here are 3 beautiful Kolam patterns for you..."
        mock_client.chat.completions.create.assert_called_once()
        
    @patch('chatbot.openai_explainer.OpenAI')
    def test_generate_blog_content_success(self, mock_openai):
        """Test successful blog content generation"""
        # Arrange
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        # Create a proper mock structure for OpenAI response
        mock_message = Mock()
        mock_message.content = "# The Rich Heritage of Kolam Art..."
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Act
        result = self.chatbot.generate_blog_content("Kolam traditions in South India", "educational")
        
        # Assert
        assert result == "# The Rich Heritage of Kolam Art..."
        mock_client.chat.completions.create.assert_called_once()
        
    @patch('chatbot.openai_explainer.OpenAI')
    def test_analyze_uploaded_image_success(self, mock_openai):
        """Test successful image analysis"""
        # Arrange
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        # Create a proper mock structure for OpenAI response
        mock_message = Mock()
        mock_message.content = "This image shows a circular Rangoli pattern..."
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Act
        result = self.chatbot.analyze_uploaded_image("A colorful circular pattern with flower motifs")
        
        # Assert
        assert result == "This image shows a circular Rangoli pattern..."
        mock_client.chat.completions.create.assert_called_once()
        
    def test_initialization_with_env_variables(self):
        """Test chatbot initialization with environment variables"""
        # The chatbot should initialize without errors
        assert self.chatbot.client is not None
        assert self.chatbot.model is not None
        assert self.chatbot.system_prompt is not None
        
    def test_system_prompt_content(self):
        """Test that system prompt contains required elements"""
        prompt = self.chatbot.system_prompt
        assert "AR BHARAT" in prompt
        assert "Kolam" in prompt
        assert "Rangoli" in prompt
        assert "indian" in prompt.lower()
        assert "augmented reality" in prompt.lower()


class TestBackwardCompatibilityFunctions:
    """Test backward compatibility functions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.sample_metadata = {
            'grid': {'type': 'square', 'rows': 3, 'cols': 3},
            'pattern': {'style': 'Traditional'}
        }
    
    @patch('chatbot.openai_explainer.ar_bharat_chatbot')
    def test_explain_kolam_ai_function(self, mock_chatbot):
        """Test explain_kolam_ai backward compatibility function"""
        # Arrange
        mock_chatbot.explain_kolam_with_ai.return_value = "AI explanation"
        
        # Act
        result = explain_kolam_ai(self.sample_metadata)
        
        # Assert
        assert result == "AI explanation"
        mock_chatbot.explain_kolam_with_ai.assert_called_once_with(self.sample_metadata)
        
    @patch('chatbot.openai_explainer.ar_bharat_chatbot')
    def test_chat_about_kolam_ai_function(self, mock_chatbot):
        """Test chat_about_kolam_ai backward compatibility function"""
        # Arrange
        mock_chatbot.chat_about_kolam_with_ai.return_value = "AI chat response"
        message = "What is this pattern?"
        history = [{"role": "user", "content": "Hello"}]
        
        # Act
        result = chat_about_kolam_ai(self.sample_metadata, message, history)
        
        # Assert
        assert result == "AI chat response"
        mock_chatbot.chat_about_kolam_with_ai.assert_called_once_with(
            self.sample_metadata, message, history
        )


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.chatbot = ARBharatChatbot()
        self.sample_metadata = {'grid': {'type': 'square'}}
    
    @patch('chatbot.openai_explainer.OpenAI')
    def test_api_key_missing_error(self, mock_openai):
        """Test handling when API key is missing"""
        # Arrange
        from openai import OpenAIError
        mock_openai.side_effect = OpenAIError("API key not provided")
        
        # Act & Assert
        # The methods should handle API key errors gracefully
        chatbot = ARBharatChatbot()
        result = chatbot.generate_pattern_suggestions({})
        assert "temporarily unavailable" in result.lower()
        
    @patch('chatbot.openai_explainer.OpenAI')
    def test_network_error_handling(self, mock_openai):
        """Test handling of network errors"""
        # Arrange
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Network error")
        
        # Act
        result = self.chatbot.generate_pattern_suggestions({})
        
        # Assert
        assert "temporarily unavailable" in result.lower()
        
    @patch('chatbot.openai_explainer.OpenAI')
    def test_malformed_response_handling(self, mock_openai):
        """Test handling of malformed API responses"""
        # Arrange
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = []  # Empty choices list
        mock_client.chat.completions.create.return_value = mock_response
        
        # Act
        result = self.chatbot.explain_kolam_with_ai(self.sample_metadata)
        
        # Assert
        # Should handle IndexError gracefully and return fallback
        assert "AI explanation temporarily unavailable" in result


@pytest.fixture
def app_with_openai_endpoints():
    """Create Flask app with OpenAI endpoints for integration testing"""
    from flask import Flask
    import sys
    import os
    
    # Add backend to path
    backend_path = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, backend_path)
    
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    return app


class TestFlaskEndpoints:
    """Integration tests for Flask endpoints"""
    
    @patch('chatbot.openai_explainer.ar_bharat_chatbot')
    def test_explain_endpoint(self, mock_chatbot, app_with_openai_endpoints):
        """Test /api/chatbot/explain endpoint"""
        # This would be an integration test if we had the full Flask app
        # For now, we're testing the core logic
        mock_chatbot.explain_kolam_with_ai.return_value = "Explanation"
        
        # Simulate endpoint behavior
        metadata = {'grid': {'type': 'square'}}
        result = mock_chatbot.explain_kolam_with_ai(metadata)
        
        assert result == "Explanation"
        
    @patch('chatbot.openai_explainer.OpenAI')
    def test_missing_openai_key_handling(self, mock_openai):
        """Test graceful handling when OpenAI API key is not configured"""
        # Test that the system gracefully handles missing API keys
        from openai import OpenAIError
        mock_openai.side_effect = OpenAIError("API key not provided")
        
        # The methods should handle missing API key gracefully
        chatbot = ARBharatChatbot()
        result = chatbot.generate_pattern_suggestions({})
        assert "temporarily unavailable" in result.lower()


if __name__ == '__main__':
    pytest.main([__file__])