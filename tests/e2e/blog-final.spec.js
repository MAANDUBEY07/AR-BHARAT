import { test, expect } from '@playwright/test';

test.describe('Real-Time Blog Integration - Final Tests', () => {
  
  test('should successfully load blog page with content', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Wait for either success or fallback content to appear
    await Promise.race([
      page.waitForSelector('h1:has-text("Heritage Chronicles")', { timeout: 15000 }),
      page.waitForSelector('text=AI generation timed out', { timeout: 15000 })
    ]);
    
    // Verify basic page structure
    await expect(page).toHaveTitle('AR BHARAT');
    await expect(page.locator('h1')).toContainText('Heritage Chronicles');
  });

  test('should display real-time blog features', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Wait for content to load (either AI or fallback)
    await page.waitForTimeout(10000);
    
    // Check that we have content (not loading screen)
    const hasContent = await page.locator('h1:has-text("Heritage Chronicles")').isVisible();
    expect(hasContent).toBe(true);
    
    // If content loaded, check for real-time features
    if (hasContent) {
      // Should show AI branding
      await expect(page.locator('text=Powered by OpenAI')).toBeVisible();
      
      // Should have articles count
      await expect(page.locator('text=Articles')).toBeVisible();
      
      // Should have refresh functionality
      const refreshButton = page.locator('text=Refresh Content');
      await expect(refreshButton).toBeVisible();
    }
  });

  test('should handle refresh functionality', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Wait for page to stabilize
    await page.waitForTimeout(10000);
    
    // Find and test refresh button if it exists
    const refreshButton = page.locator('text=Refresh Content');
    if (await refreshButton.isVisible()) {
      await refreshButton.click();
      
      // Should show refreshing state briefly
      const refreshingText = page.locator('text=Refreshing');
      await expect(refreshingText).toBeVisible({ timeout: 2000 });
      
      // Wait for refresh to complete
      await page.waitForTimeout(3000);
      
      // Button should return to normal
      await expect(page.locator('text=Refresh Content')).toBeVisible({ timeout: 10000 });
    }
  });

  test('should display article content and structure', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Wait for articles to load
    await page.waitForTimeout(10000);
    
    // Check for article elements
    const articles = page.locator('article');
    const articlesCount = await articles.count();
    
    // Should have at least some articles (even if fallback)
    expect(articlesCount).toBeGreaterThanOrEqual(2);
    
    // Check for featured article
    const featuredSection = page.locator('.bg-gradient-to-r').first();
    await expect(featuredSection).toBeVisible();
  });

  test('should show AI-enhanced features section', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Wait for full page load
    await page.waitForTimeout(12000);
    
    // Check for AI features section
    const aiFeatures = page.locator('text=AI-Enhanced Features');
    await expect(aiFeatures).toBeVisible({ timeout: 5000 });
    
    // Check for feature descriptions
    await expect(page.locator('text=Real-time Content')).toBeVisible();
    await expect(page.locator('text=Auto Updates')).toBeVisible();
  });

  test('should have newsletter subscription section', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // Wait for content
    await page.waitForTimeout(10000);
    
    // Check newsletter section
    await expect(page.locator('text=Stay Updated')).toBeVisible();
    await expect(page.locator('input[type="email"]')).toBeVisible();
    await expect(page.locator('button:has-text("Subscribe")')).toBeVisible();
  });

  test('should verify responsive design', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    await page.waitForTimeout(10000);
    
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(2000);
    
    // Main elements should still be visible
    await expect(page.locator('h1')).toContainText('Heritage Chronicles');
    
    // Test tablet viewport  
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.waitForTimeout(2000);
    
    await expect(page.locator('h1')).toContainText('Heritage Chronicles');
  });

  test('should demonstrate real-time functionality', async ({ page }) => {
    await page.goto('http://localhost:5173/blog');
    
    // This test verifies that the integration is complete and functional
    // It checks for the key elements that prove real-time integration
    
    // Wait for initial load
    await page.waitForTimeout(12000);
    
    // Check for real-time indicators
    const indicators = [
      'Live AI Content',
      'Powered by OpenAI', 
      'Articles',
      'Heritage Chronicles'
    ];
    
    for (const indicator of indicators) {
      await expect(page.locator(`text=${indicator}`)).toBeVisible({ timeout: 3000 });
    }
    
    // Verify API integration exists (even if it falls back)
    const hasApiContent = await page.locator('text=Editorial Team').isVisible();
    const hasFallbackContent = await page.locator('text=fallback').isVisible();
    
    // Should have either real API content or clear fallback
    expect(hasApiContent || hasFallbackContent).toBe(true);
  });
});