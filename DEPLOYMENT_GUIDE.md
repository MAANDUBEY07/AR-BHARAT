# SIH Project Deployment Guide

## ⚠️ CRITICAL: CREATE NEW VERCEL PROJECT

**DO NOT redeploy existing project - create fresh one to avoid cached errors**

## 1. Frontend (Vercel) - NEW PROJECT REQUIRED
- Go to https://vercel.com/new (NOT import/existing)
- Import your GitHub repo: **AR-BHARAT**
- **Framework**: Vite *(auto-detected)*
- **Root Directory**: `.` *(leave blank - frontend is now at root)*
- Click **Deploy**
- Copy your live site URL

## 2. Backend (Render) - Option A: Python Service
- Go to https://dashboard.render.com/new/web-service
- Connect your GitHub repo (SIH)
- Set root directory to `backend`
- Set start command: `gunicorn -b 0.0.0.0:$PORT app:app --workers 2 --timeout 120`
- Ensure `requirements.txt` is present in `backend/`
- Click Deploy
- Copy your backend URL

## 2. Backend (Render) - Option B: Docker Service  
- Go to https://dashboard.render.com/new/web-service
- Connect your GitHub repo (SIH)
- Leave root directory blank (uses repository root)
- Render will auto-detect the Dockerfile
- Click Deploy
- Copy your backend URL

## 3. Update Frontend API URLs
- In your frontend code, replace any `localhost` API URLs with your Render backend URL
- Example: `https://your-backend.onrender.com/api/kolam-from-image`

## 4. Test Your Live Site
- Visit your Vercel URL
- Upload images and verify Kolam generation

---

### Notes
- **Frontend is now at repository root level** - no more `/frontend/` subdirectory issues!
- API URLs are in `src/pages/UploadPage.jsx` and other components (no `/frontend/` prefix)
- For any issues, check Render and Vercel build logs for errors.
- Your backend runs from the `backend` directory with all dependencies in `requirements.txt`

---

**Ready to deploy!**
