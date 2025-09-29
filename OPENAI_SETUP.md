# OpenAI API Key Setup Guide for AR BHARAT

## 🚀 Quick Setup Steps

### 1. Get Your OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in to your OpenAI account (create one if needed)
3. Click "Create new secret key"
4. Give it a name (e.g., "AR-BHARAT-KEY")
5. Copy the API key (it starts with `sk-`)

### 2. Configure the API Key
Edit the `.env` file in your project root:

```bash
# Before (placeholder)
OPENAI_API_KEY=your_openai_api_key_here

# After (your actual key)
OPENAI_API_KEY=sk-your-actual-openai-key-here
```

### 3. Test the Connection
Run the test script:
```bash
python test_openai_connection.py
```

## 🔧 Current Status
- ✅ OpenAI client initialization: Working
- ✅ Fallback system: Active
- ❌ API key: Not configured (using placeholder)
- ✅ Error handling: Graceful degradation

## 💰 Cost Considerations
- **Model**: gpt-4o-mini (cost-effective)
- **Estimated cost**: ~$0.001-0.01 per conversation
- **Daily usage**: Depends on user interactions

## 🛡️ Security Best Practices
- Never commit your API key to version control
- Keep the `.env` file in your `.gitignore`
- Monitor your OpenAI usage in the dashboard
- Set spending limits in your OpenAI account

## 🧪 Testing Your Setup
Once configured, the system will:
1. Use AI for rich, culturally-aware explanations
2. Provide pattern analysis and mathematical insights
3. Offer conversational chat about Indian art
4. Analyze uploaded images for better pattern conversion

## 🔄 Fallback System
Even without an API key, the system remains functional:
- Falls back to rule-based explanations
- Maintains core functionality
- No crashes or errors
- Users still get helpful information

## 📞 Support
If you encounter issues:
1. Check your API key format (starts with `sk-`)
2. Verify your OpenAI account has credits
3. Test with the provided test script
4. Check network connectivity

Happy coding! 🎨✨