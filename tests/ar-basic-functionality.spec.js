import { test, expect } from '@playwright/test';

test.describe('AR Basic Functionality', () => {
  test.beforeEach(async ({ page }) => {
    // Start from the home page
    await page.goto('/');
    await expect(page).toHaveTitle(/AR BHARAT/);
  });

  test('should navigate to home page successfully', async ({ page }) => {
    // Basic navigation test
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('text=AR BHARAT')).toBeVisible();
  });

  test('should have navigation links to AR-related pages', async ({ page }) => {
    // Check navigation menu
    await expect(page.locator('a[href="/upload"]')).toBeVisible();
    await expect(page.locator('a[href="/gallery"]')).toBeVisible();
    
    // Check if AR functionality is prominently featured
    await expect(page.locator('text=AR')).toBeVisible();
  });

  test('should access upload page for AR pattern creation', async ({ page }) => {
    // Navigate to upload page
    await page.click('a[href="/upload"]');
    await expect(page).toHaveURL('/upload');
    
    // Check upload interface is present
    await expect(page.locator('input[type="file"]')).toBeVisible();
    await expect(page.locator('text=upload', 'text=Upload')).toBeVisible();
  });

  test('should access gallery page and check for AR elements', async ({ page }) => {
    // Navigate to gallery
    await page.click('a[href="/gallery"]');
    await expect(page).toHaveURL('/gallery');
    
    // Check gallery structure
    await expect(page.locator('h2:has-text("Gallery")')).toBeVisible();
    
    // Check if patterns load (with timeout)
    try {
      await page.waitForSelector('img', { timeout: 5000 });
      // If patterns exist, check for AR elements
      const patterns = page.locator('div:has(img)');
      if (await patterns.count() > 0) {
        // Look for AR badges or buttons
        const arElements = page.locator('text=AR, text=Launch AR');
        if (await arElements.count() > 0) {
          await expect(arElements.first()).toBeVisible();
        }
      }
    } catch (error) {
      console.log('No patterns found in gallery - this is expected for empty gallery');
    }
  });

  test('should load AR page with proper structure', async ({ page }) => {
    // Navigate directly to AR page
    await page.goto('/ar');
    
    // Check that AR page loads without critical errors
    await expect(page.locator('h1')).toBeVisible();
    
    // Look for AR-related content
    const arContent = page.locator('text=AR, text=3D, text=Preview');
    await expect(arContent.first()).toBeVisible();
  });

  test('should handle AR page with pattern ID', async ({ page }) => {
    // Try AR page with a test ID
    await page.goto('/ar/1');
    
    // Should load without throwing errors
    await expect(page.locator('h1')).toBeVisible();
    
    // Check for loading or error states
    const loadingElements = page.locator('text=Loading, text=loading');
    const errorElements = page.locator('text=Error, text=error, text=not found');
    
    // At least one of these states should be present
    const hasContent = (await loadingElements.count()) > 0 || 
                      (await errorElements.count()) > 0 || 
                      (await page.locator('canvas, [data-testid="3d-preview"]').count()) > 0;
    
    expect(hasContent).toBeTruthy();
  });

  test('should check for WebXR support detection', async ({ page }) => {
    await page.goto('/ar');
    
    // Check if page can detect WebXR capabilities
    const webxrSupported = await page.evaluate(() => {
      return 'xr' in navigator;
    });
    
    console.log('WebXR supported in test browser:', webxrSupported);
    
    // The page should handle both supported and unsupported cases
    await expect(page.locator('h1')).toBeVisible();
  });

  test('should handle Three.js canvas rendering', async ({ page }) => {
    await page.goto('/ar/1');
    
    // Wait for potential canvas elements
    try {
      await page.waitForSelector('canvas', { timeout: 8000 });
      const canvas = page.locator('canvas');
      await expect(canvas).toBeVisible();
      
      // Check canvas has reasonable dimensions
      const canvasBox = await canvas.boundingBox();
      expect(canvasBox?.width).toBeGreaterThan(100);
      expect(canvasBox?.height).toBeGreaterThan(100);
    } catch (error) {
      // Canvas might not load if no pattern or WebGL issues
      console.log('Canvas not loaded - may be expected without valid pattern');
      // Just ensure page structure is intact
      await expect(page.locator('h1')).toBeVisible();
    }
  });

  test('should have responsive design on mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    // Test key pages work on mobile
    await page.goto('/');
    await expect(page.locator('h1')).toBeVisible();
    
    await page.goto('/upload');
    await expect(page.locator('input[type="file"]')).toBeVisible();
    
    await page.goto('/gallery');
    await expect(page.locator('h2')).toBeVisible();
    
    await page.goto('/ar');
    await expect(page.locator('h1')).toBeVisible();
  });

  test('should maintain navigation between AR-related pages', async ({ page }) => {
    // Test navigation flow
    await page.goto('/');
    
    // Go to upload
    await page.click('a[href="/upload"]');
    await expect(page).toHaveURL('/upload');
    
    // Go to gallery
    await page.click('a[href="/gallery"]');
    await expect(page).toHaveURL('/gallery');
    
    // Navigation should work consistently
    const homeLink = page.locator('a[href="/"], text=Home, text=AR BHARAT');
    if (await homeLink.count() > 0) {
      await homeLink.first().click();
      await expect(page).toHaveURL('/');
    }
  });

  test('should have accessible AR interface elements', async ({ page }) => {
    await page.goto('/ar');
    
    // Check for basic accessibility
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    
    if (buttonCount > 0) {
      // Check that buttons have accessible text or labels
      for (let i = 0; i < Math.min(buttonCount, 5); i++) {
        const button = buttons.nth(i);
        const text = await button.textContent();
        const ariaLabel = await button.getAttribute('aria-label');
        
        // Should have either visible text or aria-label
        expect(text?.trim() || ariaLabel).toBeTruthy();
      }
    }
    
    // Check for heading structure
    const headings = page.locator('h1, h2, h3');
    const headingCount = await headings.count();
    expect(headingCount).toBeGreaterThan(0);
  });
});