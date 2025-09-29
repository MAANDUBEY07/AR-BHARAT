"""
OpenAI-powered chatbot for AR BHARAT - Kolam and Indian cultural art explanation
"""

import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ARBharatChatbot:
    def __init__(self, client=None):
        if client is not None:
            # Allow dependency injection for testing
            self.client = client
        else:
            try:
                self.client = OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY')
                )
            except Exception:
                # If OpenAI client initialization fails, set to None
                # Methods will handle this gracefully
                self.client = None
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.system_prompt = """
        You are an AI assistant for AR BHARAT, a platform that transforms traditional Indian Rangoli and Kolam patterns into stunning augmented reality art experiences. You specialize in:

        1. **Indian Cultural Heritage**: Expert knowledge about Kolam, Rangoli, Mandala, and other traditional Indian art forms
        2. **Mathematical Patterns**: Understanding geometric principles, symmetry, and mathematical concepts in Indian art
        3. **AR Technology**: How augmented reality enhances traditional art experiences
        4. **Art Creation**: Guiding users through the process of creating and converting patterns

        **Your Personality**: Knowledgeable, enthusiastic about Indian culture, educational yet approachable, and technically informed about both art and technology.

        **Response Style**: 
        - Provide rich cultural context when discussing patterns
        - Explain mathematical concepts in simple terms
        - Connect traditional art to modern technology
        - Be encouraging and supportive of users' creative endeavors
        - Include relevant emoji when appropriate 🎨🇮🇳✨
        """

    def explain_kolam_with_ai(self, meta: Dict[str, Any]) -> str:
        """Generate AI-powered explanation of a Kolam pattern"""
        try:
            if self.client is None:
                raise Exception("OpenAI client not available")
            grid = meta.get('grid', {})
            symmetry = meta.get('symmetry', {})
            pattern = meta.get('pattern', {})
            
            prompt = f"""
            Explain this Kolam pattern in a culturally rich and educational way:
            
            Grid Details:
            - Type: {grid.get('type', 'square')}
            - Dimensions: {grid.get('rows')} × {grid.get('cols')}
            - Spacing: {grid.get('spacing')} units
            
            Symmetry:
            - Rotational symmetry: {symmetry.get('rotational', 'Unknown')}
            - Reflection axes: {symmetry.get('reflection_axes', 'Unknown')}
            
            Pattern:
            - Style: {pattern.get('style', 'Traditional')}
            - Complexity: {pattern.get('complexity', 'Medium')}
            
            Please provide:
            1. Cultural significance and regional context
            2. Mathematical properties and geometric principles
            3. Traditional creation methods
            4. Spiritual or ceremonial meaning
            5. How this fits into India's art heritage
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            # Check if response has choices and content
            if not response.choices or not response.choices[0].message.content:
                raise Exception("Empty response from OpenAI")
                
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback to original explainer if OpenAI fails
            from .explainer import explain_kolam
            return f"🎨 {explain_kolam(meta)} (AI explanation temporarily unavailable)"

    def chat_about_kolam_with_ai(self, meta: Dict[str, Any], message: str, conversation_history: Optional[list] = None) -> str:
        """AI-powered conversational chat about Kolam patterns"""
        try:
            if self.client is None:
                raise Exception("OpenAI client not available")
            # Build context about the current pattern
            context = f"""
            Current Kolam Pattern Context:
            - Grid: {meta.get('grid', {}).get('type', 'square')} {meta.get('grid', {}).get('rows')}×{meta.get('grid', {}).get('cols')}
            - Style: {meta.get('pattern', {}).get('style', 'Traditional')}
            - Symmetry: {meta.get('symmetry', {}).get('rotational', 'Unknown')}-fold rotational
            - Region: {meta.get('region_hint', 'South India')}
            """
            
            # Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": context}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                # Convert frontend message format to OpenAI format
                for msg in conversation_history[-6:]:
                    if isinstance(msg, dict) and 'content' in msg:
                        if msg.get('type') == 'user':
                            messages.append({"role": "user", "content": msg['content']})
                        elif msg.get('type') == 'bot':
                            messages.append({"role": "assistant", "content": msg['content']})
            
            messages.append({"role": "user", "content": message})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=300,
                temperature=0.8
            )
            
            # Check if response has choices and content
            if not response.choices or not response.choices[0].message.content:
                raise Exception("Empty response from OpenAI")
                
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback to rule-based chat
            from .explainer import chat_about_kolam
            return f"🤖 {chat_about_kolam(meta, message)} (AI chat temporarily unavailable)"

    def generate_pattern_suggestions(self, user_preferences: Dict[str, Any]) -> str:
        """Generate AI-powered pattern suggestions based on user preferences"""
        try:
            if self.client is None:
                raise Exception("OpenAI client not available")
            prompt = f"""
            A user wants to create Kolam/Rangoli patterns with these preferences:
            - Difficulty level: {user_preferences.get('difficulty', 'Medium')}
            - Cultural region: {user_preferences.get('region', 'Any')}
            - Occasion: {user_preferences.get('occasion', 'Daily practice')}
            - Size preference: {user_preferences.get('size', 'Medium')}
            - Color preferences: {user_preferences.get('colors', 'Traditional')}
            
            Suggest 3 specific Kolam/Rangoli patterns with:
            1. Pattern name and cultural background
            2. Symbolic meaning
            3. Step-by-step creation tips
            4. Modern AR enhancement ideas
            
            Make it inspiring and educational! 🎨
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.9
            )
            
            # Check if response has choices and content
            if not response.choices or not response.choices[0].message.content:
                raise Exception("Empty response from OpenAI")
                
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return "🎨 Pattern suggestions are temporarily unavailable. Please try again later!"

    def generate_blog_content(self, topic: str, style: str = "educational") -> str:
        """Generate AI-powered blog content about Indian art and culture"""
        try:
            if self.client is None:
                raise Exception("OpenAI client not available")
            style_prompts = {
                "educational": "Write in an educational, informative style suitable for students and art enthusiasts",
                "storytelling": "Write in a storytelling style that brings the culture to life with narratives",
                "technical": "Write in a technical style focusing on methods, techniques, and processes",
                "cultural": "Write with deep cultural context and traditional perspectives"
            }
            
            prompt = f"""
            Write a blog post about: {topic}
            
            Style: {style_prompts.get(style, style_prompts['educational'])}
            
            Include:
            1. Engaging introduction
            2. Historical context
            3. Cultural significance
            4. Modern relevance and AR technology connections
            5. Call to action for readers to engage with AR BHARAT
            
            Make it approximately 500-700 words, engaging, and SEO-friendly.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            # Check if response has choices and content
            if not response.choices or not response.choices[0].message.content:
                raise Exception("Empty response from OpenAI")
                
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Blog content generation temporarily unavailable for topic: {topic}"

    def analyze_uploaded_image(self, image_description: str) -> str:
        """Provide AI analysis of uploaded images for better pattern conversion"""
        try:
            if self.client is None:
                raise Exception("OpenAI client not available")
            prompt = f"""
            A user uploaded an image for Kolam/Rangoli conversion with this description:
            {image_description}
            
            Provide helpful analysis including:
            1. Suggested grid size and type
            2. Recommended processing parameters
            3. Expected pattern characteristics
            4. Tips for best AR experience
            5. Cultural context if recognizable patterns are present
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.6
            )
            
            # Check if response has choices and content
            if not response.choices or not response.choices[0].message.content:
                raise Exception("Empty response from OpenAI")
                
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return "Image analysis temporarily unavailable. Please proceed with standard conversion."

# Initialize global chatbot instance
ar_bharat_chatbot = ARBharatChatbot()

# Backward compatibility functions
def explain_kolam_ai(meta: Dict[str, Any]) -> str:
    """Backward compatible function for AI-powered Kolam explanation"""
    return ar_bharat_chatbot.explain_kolam_with_ai(meta)

def chat_about_kolam_ai(meta: Dict[str, Any], message: str, history: Optional[list] = None) -> str:
    """Backward compatible function for AI-powered chat"""
    return ar_bharat_chatbot.chat_about_kolam_with_ai(meta, message, history)