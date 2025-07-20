# Render Deployment Guide

## Quick Deployment Steps

### 1. Push to Git Repository
Make sure your code is in a Git repository (GitHub, GitLab, etc.)

### 2. Deploy to Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your Git repository
4. Configure the service:
   - **Name**: `better-perplexity-backend`
   - **Root Directory**: `backend/` (if your repo has frontend/backend structure)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### 3. Set Environment Variables

In the Render dashboard, go to **Environment** tab and add:

- `BRAVE_API_KEY` = your Brave Search API key
- `OPENAI_API_KEY` = your OpenAI API key
- `RERANKER_MODEL` = `jinaai/jina-reranker-v1-turbo-en`
- `TRAINING_S3_BUCKET` = your S3 bucket name (optional)

### 4. Deploy

Click **"Create Web Service"** and wait for deployment.

### 5. Get Your Backend URL

Once deployed, you'll get a URL like:
`https://better-perplexity-backend.onrender.com`

### 6. Update Frontend Environment

1. Go to your Vercel dashboard
2. Select your frontend project
3. Go to **Settings** → **Environment Variables**
4. Update `VITE_API_BASE_URL` to your Render backend URL

## Alternative: Using Dockerfile

If you prefer to use the Dockerfile:

1. In Render, select **"Docker"** instead of **"Python"**
2. Render will automatically use the `Dockerfile`
3. No need to specify build/start commands

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `BRAVE_API_KEY` | Yes | Your Brave Search API key |
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `RERANKER_MODEL` | No | Reranker model ID (default: jinaai/jina-reranker-v1-turbo-en) |
| `TRAINING_S3_BUCKET` | No | S3 bucket for training data persistence |

## Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check that `requirements.txt` is in the backend directory
   - Ensure all dependencies are listed

2. **Import Errors**:
   - Make sure all Python files are in the correct location
   - Check that `storage.py` is in the backend directory

3. **Environment Variables**:
   - Verify all required environment variables are set
   - Check that API keys are valid

4. **CORS Issues**:
   - Update `ALLOWED_ORIGINS` in `app.py` with your frontend URL
   - Redeploy after making changes

### Performance Considerations

1. **Cold Starts**: The first request may be slow due to model loading
2. **Memory Usage**: Monitor memory usage in Render dashboard
3. **Timeout**: Requests may timeout if models take too long to load

## Monitoring

- Check Render dashboard for logs and performance metrics
- Monitor API usage and costs
- Set up alerts for errors

## Scaling

- Start with the **Starter** plan
- Upgrade to **Standard** or **Pro** if you need more resources
- Consider using **Autoscaling** for high traffic 