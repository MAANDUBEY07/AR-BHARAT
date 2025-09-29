# SIH Project Deployment Guide

## ⚠️ CRITICAL: CREATE NEW VERCEL PROJECT

**DO NOT redeploy existing project - create fresh one to avoid cached errors**

## 🔧 ENVIRONMENT VARIABLES SETUP

### For Vercel (Frontend)
After creating your project, go to **Project Settings → Environment Variables** and add:

| Variable Name | Value | Description |
|---------------|--------|-------------|
| `VITE_API_URL` | `https://your-backend.onrender.com` | Your Render backend URL |
| `NODE_ENV` | `production` | Production environment flag |

### For Render (Backend)
In your Render service settings, go to **Environment** and add:

| Variable Name | Value | Description |
|---------------|--------|-------------|
| `OPENAI_API_KEY` | `your-openai-api-key` | OpenAI API key for chatbot (get from [OpenAI Dashboard](https://platform.openai.com/api-keys)) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `FLASK_ENV` | `production` | Flask environment |
| `DEBUG` | `False` | Disable debug mode in production |

#### 🔑 Getting Your OpenAI API Key:
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in/create account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-proj-...`)
5. **⚠️ Important:** Add billing/payment method to OpenAI account (required for API usage)
6. The chatbot will gracefully fallback to rule-based responses if no API key is provided

## 1. Frontend (Vercel) - NEW PROJECT REQUIRED
- Go to https://vercel.com/new (NOT import/existing)
- Import your GitHub repo: **AR-BHARAT**
- **Framework**: Vite *(auto-detected)*
- **Root Directory**: `.` *(leave blank - frontend is now at root)*
- **Build Command**: `npm run build` *(should auto-detect)*
- **Output Directory**: `dist` *(should auto-detect)*
- **⚠️ BEFORE DEPLOYING:** Add environment variables (see above)
- Click **Deploy**
- Copy your live site URL

## 2. Backend (Render) - Option A: Python Service
- Go to https://dashboard.render.com/new/web-service
- Connect your GitHub repo (SIH)
- Set root directory to `backend`
- Set start command: `gunicorn -b 0.0.0.0:$PORT app:app --workers 2 --timeout 120`
- Ensure `requirements.txt` is present in `backend/`
- **⚠️ BEFORE DEPLOYING:** Add environment variables (see above table)
- Click Deploy
- Copy your backend URL

## 2. Backend (Render) - Option B: Docker Service  
- Go to https://dashboard.render.com/new/web-service
- Connect your GitHub repo (SIH)
- Leave root directory blank (uses repository root)
- Render will auto-detect the Dockerfile
- **⚠️ BEFORE DEPLOYING:** Add environment variables (see above table)
- Click Deploy
- Copy your backend URL

## 3. Update Environment Variables After Backend Deployment
1. **Copy your Render backend URL** (e.g., `https://ar-bharat-1.onrender.com`)
2. **Go back to Vercel → Project Settings → Environment Variables**
3. **Update `VITE_API_URL`** with your actual backend URL
4. **Redeploy your Vercel project** for changes to take effect

## 4. Test Your Live Site
- Visit your Vercel URL
- Upload images and verify Kolam generation
- Test chatbot functionality (requires OpenAI API key)

---

### 🎯 What Each Environment Variable Enables

| Variable | Feature | Impact if Missing |
|----------|---------|------------------|
| `VITE_API_URL` | Frontend-Backend Communication | ❌ Image upload fails, no Kolam generation |
| `OPENAI_API_KEY` | AI-Powered Chatbot | 🔄 Falls back to rule-based responses |
| `OPENAI_MODEL` | AI Model Selection | 🔄 Uses default model |
| `FLASK_ENV` | Backend Environment | 🔧 Uses development settings |
| `DEBUG` | Debug Mode | 🔧 May expose sensitive info in production |

### 📝 Notes
- **Frontend is now at repository root level** - no more `/frontend/` subdirectory issues!
- **API URLs auto-configure** based on `VITE_API_URL` environment variable
- **OpenAI integration is optional** - app works without it (chatbot uses fallback)
- **Check build logs** in both Vercel and Render if deployment fails
- **Backend runs from `backend/` directory** with all dependencies in `requirements.txt`

---

**🚀 Ready to deploy with full environment variable support!**

---

## 📋 QUICK DEPLOYMENT CHECKLIST

### Before You Start:
- [ ] Get OpenAI API key (optional but recommended)
- [ ] Have your GitHub repo ready
- [ ] Know your project name (lowercase, no spaces)

### Deployment Steps:
1. [ ] **Deploy Backend on Render** with environment variables
2. [ ] **Copy backend URL** (e.g., `https://ar-bharat-1.onrender.com`)
3. [ ] **Deploy Frontend on Vercel** with `VITE_API_URL` set to backend URL
4. [ ] **Test the deployment** by uploading an image
5. [ ] **Optional:** Test chatbot with OpenAI integration

### If Something Goes Wrong:

#### Build Errors (Vercel):
- **"vite: command not found"**: 
  - Ensure Build Command is set to `npm run build` (NOT `vite build`)
  - Verify `vercel.json` is present in root directory
  - Check that `vite` is in devDependencies in package.json
- **"Cannot find package 'vite'"**: Update vite to version 5.x in package.json and delete package-lock.json, then redeploy
- **"Node version mismatch"**: Set Node version to 18.x in project settings  
- **"Build command failed"**: Verify build script is `"build": "vite build"` in package.json
- **"Output directory not found"**: Ensure output directory is set to `dist`

#### General Issues:
- Check build logs in both Vercel and Render dashboards
- Verify environment variables are set correctly
- Ensure backend URL doesn't have trailing slashes
- Try redeploying after fixing environment variables
- Clear deployment cache and try again

### Example URLs After Deployment:
- Frontend: `https://your-project.vercel.app`
- Backend: `https://ar-bharat-1.onrender.com`
- API Endpoint: `https://ar-bharat-1.onrender.com/api/kolam-from-image`
