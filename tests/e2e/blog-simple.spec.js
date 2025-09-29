import { test, expect } from '@playwright/test';

// Simple tests for real-time blog integration
test.describe('Real-Time Blog - Core Features', () => {
  
  test('should load blog page successfully', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Wait for main content to appear
    await expect(page.locator('h1')).toContainText('Heritage Chronicles');
    await expect(page).toHaveTitle('AR BHARAT');
  });

  test('should display live AI content indicator', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Check for live indicator
    await expect(page.locator('text=Live AI Content')).toBeVisible();
    
    // Check OpenAI branding
    await expect(page.locator('text=🤖 Powered by OpenAI')).toBeVisible();
  });

  test('should display articles and refresh button', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Check featured article is present
    const featuredArticle = page.locator('.bg-gradient-to-r').first();
    await expect(featuredArticle).toBeVisible();
    
    // Check regular articles
    await expect(page.locator('article')).toHaveCount(5);
    
    // Check refresh button
    const refreshButton = page.locator('text=🔄 Refresh Content');
    await expect(refreshButton).toBeVisible();
  });

  test('should handle refresh functionality', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Click refresh button
    const refreshButton = page.locator('text=🔄 Refresh Content');
    await refreshButton.click();
    
    // Should show loading state
    await expect(page.locator('text=Refreshing...')).toBeVisible();
    
    // Wait for completion and check button is re-enabled
    await page.waitForTimeout(2000);
    await expect(page.locator('text=🔄 Refresh Content')).toBeVisible();
  });

  test('should display AI features section', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Check AI features section
    await expect(page.locator('text=🤖 AI-Enhanced Features')).toBeVisible();
    await expect(page.locator('text=Real-time Content')).toBeVisible();
    await expect(page.locator('text=Trending Topics')).toBeVisible();
    await expect(page.locator('text=Auto Updates')).toBeVisible();
  });

  test('should show newsletter subscription', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Check newsletter section
    await expect(page.locator('text=Stay Updated with AI-Powered Heritage Tech')).toBeVisible();
    await expect(page.locator('input[placeholder="Enter your email"]')).toBeVisible();
    await expect(page.locator('button:has-text("Subscribe")')).toBeVisible();
  });

  test('should have working article links', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Check "Read Full Article" button exists
    await expect(page.locator('button:has-text("Read Full Article")')).toBeVisible();
    
    // Check "Read more" links exist
    await expect(page.locator('text=Read more →').first()).toBeVisible();
  });

  test('should be responsive', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    // Main elements should still be visible
    await expect(page.locator('h1')).toContainText('Heritage Chronicles');
    await expect(page.locator('text=Live AI Content')).toBeVisible();
    
    // Reset to desktop
    await page.setViewportSize({ width: 1280, height: 720 });
  });

  test('should verify backend API integration', async ({ page }) => {
    // Monitor network requests
    const apiResponsePromise = page.waitForResponse('**/api/blog/articles*');
    
    await page.goto('http://localhost:5173/blog');
    
    // Verify API call was made
    const apiResponse = await apiResponsePromise;
    expect(apiResponse.status()).toBe(200);
  });
});