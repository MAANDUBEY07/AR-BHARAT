"""
Fixed unit tests for OpenAI-powered chatbot functionality
"""

import pytest
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from unittest.mock import Mock, patch, MagicMock
from chatbot.openai_explainer import ARBharatChatbot


class TestARBharatChatbotFixed:
    """Fixed test cases for ARBharatChatbot class"""
    
    def setup_method(self):
        """Set up test fixtures"""
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
        mock_message = Mock()
        mock_message.content = "This is a beautiful traditional Kolam pattern..."
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create chatbot with mocked client
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

    def test_generate_pattern_suggestions_success(self):
        """Test successful pattern suggestions generation"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Here are 3 beautiful pattern suggestions..."
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create chatbot with mocked client
        chatbot = ARBharatChatbot(client=mock_client)
        
        # Act
        preferences = {
            'difficulty': 'Medium',
            'region': 'Tamil Nadu',
            'occasion': 'Festival'
        }
        result = chatbot.generate_pattern_suggestions(preferences)
        
        # Assert
        assert result == "Here are 3 beautiful pattern suggestions..."
        mock_client.chat.completions.create.assert_called_once()

    def test_malformed_response_handling(self):
        """Test handling of malformed API responses"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = []  # Empty choices list
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create chatbot with mocked client
        chatbot = ARBharatChatbot(client=mock_client)
        
        # Act
        result = chatbot.explain_kolam_with_ai(self.sample_metadata)
        
        # Assert - Should handle IndexError gracefully and return fallback
        assert "AI explanation temporarily unavailable" in result

    def test_empty_content_handling(self):
        """Test handling of empty content in API responses"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = None  # Empty content
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create chatbot with mocked client
        chatbot = ARBharatChatbot(client=mock_client)
        
        # Act
        result = chatbot.explain_kolam_with_ai(self.sample_metadata)
        
        # Assert - Should handle empty content and return fallback
        assert "AI explanation temporarily unavailable" in result

    def test_initialization_with_dependency_injection(self):
        """Test chatbot initialization with dependency injection"""
        # Arrange
        mock_client = Mock()
        
        # Act
        chatbot = ARBharatChatbot(client=mock_client)
        
        # Assert
        assert chatbot.client == mock_client
        
    def test_initialization_without_client(self):
        """Test chatbot initialization without providing client"""
        # Act & Assert - Should not raise exception
        chatbot = ARBharatChatbot()
        
        # The client should be None if no OpenAI API key is available
        # or should be initialized if API key is available
        assert chatbot is not None