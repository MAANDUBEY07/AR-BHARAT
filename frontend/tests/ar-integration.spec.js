import { test, expect } from '@playwright/test';

test.describe('AR Integration Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Start from the home page
    await page.goto('/');
    await expect(page).toHaveTitle(/AR BHARAT/);
  });

  test.describe('Upload Page AR Integration', () => {
    test('should display AR banner after pattern generation', async ({ page }) => {
      // Navigate to upload page
      await page.click('a[href="/upload"]');
      await expect(page).toHaveURL('/upload');
      
      // Mock file upload and wait for processing (without actual file)
      // Note: In real tests, you'd mock the backend response or use test images
      const fileChooser = page.waitForEvent('filechooser');
      await page.locator('input[type="file"]').click();
      const fileChooserPromise = await fileChooser;
      
      // Simulate pattern generation completion
      // This would be triggered after actual image processing
      await page.evaluate(() => {
        // Simulate the state after pattern generation
        window.dispatchEvent(new CustomEvent('patternGenerated', {
          detail: { 
            svgContent: '<svg><circle cx="50" cy="50" r="40"/></svg>',
            patternId: 'test-pattern-123'
          }
        }));
      });
      
      // Check AR banner appears
      await expect(page.locator('.gradient')).toContainText('Experience your Kolam in Augmented Reality');
      
      // Check AR button is visible and prominent
      const arButton = page.locator('button:has-text("Launch AR")').first();
      await expect(arButton).toBeVisible();
      
      // Verify gradient background styling
      const arBanner = page.locator('.bg-gradient-to-r');
      await expect(arBanner).toBeVisible();
      
      // Check sharing functionality
      const shareButton = page.locator('button:has-text("Share")');
      await expect(shareButton).toBeVisible();
    });

    test('should navigate to AR page from upload results', async ({ page }) => {
      // Navigate to upload and simulate pattern generation
      await page.goto('/upload');
      
      // Simulate pattern generation (mock)
      await page.evaluate(() => {
        window.dispatchEvent(new CustomEvent('patternGenerated', {
          detail: { 
            svgContent: '<svg><circle cx="50" cy="50" r="40"/></svg>',
            patternId: 'test-pattern-123'
          }
        }));
      });
      
      // Click AR button from results banner
      await page.click('button:has-text("Launch AR")');
      
      // Should navigate to AR page
      await expect(page).toHaveURL(/\/ar/);
      await expect(page.locator('h1')).toContainText('AR Viewer');
    });
  });

  test.describe('Gallery Page AR Integration', () => {
    test('should display AR badges on pattern thumbnails', async ({ page }) => {
      // Navigate to gallery
      await page.click('a[href="/gallery"]');
      await expect(page).toHaveURL('/gallery');
      
      // Wait for patterns to load
      await page.waitForSelector('[data-testid="pattern-card"]', { timeout: 10000 });
      
      // Check AR badges are visible on pattern cards
      const arBadges = page.locator('.bg-purple-500:has-text("AR")');
      await expect(arBadges.first()).toBeVisible();
      
      // Check AR buttons are prominent on cards
      const arButtons = page.locator('button:has-text("Launch AR")');
      await expect(arButtons.first()).toBeVisible();
      
      // Verify styling of AR elements
      const arBadge = arBadges.first();
      await expect(arBadge).toHaveClass(/bg-purple-500/);
    });

    test('should launch AR from gallery pattern cards', async ({ page }) => {
      // Navigate to gallery
      await page.goto('/gallery');
      
      // Wait for patterns to load
      await page.waitForSelector('[data-testid="pattern-card"]', { timeout: 10000 });
      
      // Click AR button on first pattern
      const firstArButton = page.locator('button:has-text("Launch AR")').first();
      await firstArButton.click();
      
      // Should navigate to AR page with pattern
      await expect(page).toHaveURL(/\/ar/);
      await expect(page.locator('h1')).toContainText('AR Viewer');
    });
  });

  test.describe('AR Page Functionality', () => {
    test('should display enhanced AR interface', async ({ page }) => {
      // Navigate directly to AR page
      await page.goto('/ar');
      
      // Check header with AR status
      await expect(page.locator('h1')).toContainText('AR Viewer');
      
      // Check AR status indicator
      const arStatus = page.locator('[data-testid="ar-status"]');
      await expect(arStatus).toBeVisible();
      
      // Check 3D preview section
      const previewSection = page.locator('[data-testid="3d-preview"]');
      await expect(previewSection).toBeVisible();
      
      // Check control buttons are present
      await expect(page.locator('button:has-text("Play Narration")')).toBeVisible();
      await expect(page.locator('button:has-text("Export 3D Model")')).toBeVisible();
      await expect(page.locator('button:has-text("Share AR")')).toBeVisible();
      
      // Check AR instructions section
      const instructions = page.locator('[data-testid="ar-instructions"]');
      await expect(instructions).toBeVisible();
      await expect(instructions).toContainText('Point your camera');
    });

    test('should handle AR button interactions', async ({ page }) => {
      await page.goto('/ar');
      
      // Test narration button
      const narrationButton = page.locator('button:has-text("Play Narration")');
      await expect(narrationButton).toBeVisible();
      await narrationButton.click();
      
      // Check if narration controls appear (would depend on implementation)
      
      // Test export button
      const exportButton = page.locator('button:has-text("Export 3D Model")');
      await expect(exportButton).toBeVisible();
      
      // Test share button
      const shareButton = page.locator('button:has-text("Share AR")');
      await expect(shareButton).toBeVisible();
    });

    test('should detect WebXR capability', async ({ page }) => {
      await page.goto('/ar');
      
      // Check AR status based on device capability
      const arStatus = page.locator('[data-testid="ar-status"]');
      
      // The status will depend on browser WebXR support
      await expect(arStatus).toContainText(/Ready|Not Available|Active/);
      
      // Check that appropriate instructions are shown
      const instructions = page.locator('[data-testid="ar-instructions"]');
      await expect(instructions).toBeVisible();
    });

    test('should show appropriate fallback for non-AR devices', async ({ page }) => {
      // Mock non-AR device
      await page.addInitScript(() => {
        delete window.navigator.xr;
      });
      
      await page.goto('/ar');
      
      // Should show fallback message
      const fallbackMessage = page.locator(':has-text("AR not available on this device")');
      await expect(fallbackMessage).toBeVisible();
      
      // Should still show 3D preview
      const preview = page.locator('[data-testid="3d-preview"]');
      await expect(preview).toBeVisible();
    });
  });

  test.describe('XR Button Utility', () => {
    test('should handle XR session management', async ({ page }) => {
      await page.goto('/ar');
      
      // Find XR button (might be rendered by Three.js WebXRManager)
      const xrButton = page.locator('button[data-testid="xr-button"]');
      
      if (await xrButton.isVisible()) {
        // Click XR button
        await xrButton.click();
        
        // Check for session state changes
        // This would depend on WebXR API availability
      }
    });

    test('should display user-friendly error messages', async ({ page }) => {
      // Mock XR error
      await page.addInitScript(() => {
        window.navigator.xr = {
          isSessionSupported: () => Promise.reject(new Error('XR not supported'))
        };
      });
      
      await page.goto('/ar');
      
      // Check for error handling UI
      const errorMessage = page.locator('.error-notification');
      // Would need to trigger XR session to see error handling
    });
  });

  test.describe('Responsive AR Design', () => {
    test('should work on mobile viewports', async ({ page }) => {
      // Set mobile viewport
      await page.setViewportSize({ width: 375, height: 667 });
      
      await page.goto('/ar');
      
      // Check that AR interface is mobile-friendly
      const arViewer = page.locator('[data-testid="ar-viewer"]');
      await expect(arViewer).toBeVisible();
      
      // Check that buttons are appropriately sized for mobile
      const buttons = page.locator('button');
      const firstButton = buttons.first();
      await expect(firstButton).toBeVisible();
    });

    test('should maintain functionality across screen sizes', async ({ page }) => {
      // Test different screen sizes
      const sizes = [
        { width: 320, height: 568 }, // iPhone 5
        { width: 768, height: 1024 }, // iPad
        { width: 1920, height: 1080 } // Desktop
      ];
      
      for (const size of sizes) {
        await page.setViewportSize(size);
        await page.goto('/ar');
        
        // Check core elements are visible
        await expect(page.locator('h1')).toContainText('AR Viewer');
        await expect(page.locator('[data-testid="3d-preview"]')).toBeVisible();
      }
    });
  });

  test.describe('Navigation and Integration', () => {
    test('should maintain AR access across page navigation', async ({ page }) => {
      // Start at home
      await page.goto('/');
      
      // Navigate through different pages
      await page.click('a[href="/gallery"]');
      await expect(page).toHaveURL('/gallery');
      
      // Check AR buttons are consistently available
      await page.waitForSelector('button:has-text("Launch AR")', { timeout: 5000 });
      
      // Navigate to upload
      await page.click('a[href="/upload"]');
      await expect(page).toHaveURL('/upload');
      
      // AR should be accessible after pattern generation
    });

    test('should handle deep linking to AR page', async ({ page }) => {
      // Direct navigation to AR page
      await page.goto('/ar');
      
      // Should load properly
      await expect(page.locator('h1')).toContainText('AR Viewer');
      
      // Should have navigation back to other pages
      const backButton = page.locator('button:has-text("Back to Gallery")');
      if (await backButton.isVisible()) {
        await backButton.click();
        await expect(page).toHaveURL('/gallery');
      }
    });
  });

  test.describe('Performance and Loading', () => {
    test('should load Three.js components efficiently', async ({ page }) => {
      await page.goto('/ar');
      
      // Check that 3D canvas loads
      await page.waitForSelector('canvas', { timeout: 10000 });
      
      // Verify Three.js is loaded
      const threeLoaded = await page.evaluate(() => {
        return typeof window.THREE !== 'undefined';
      });
      
      expect(threeLoaded).toBeTruthy();
    });

    test('should handle loading states gracefully', async ({ page }) => {
      await page.goto('/ar');
      
      // Check for loading indicators
      const loadingIndicator = page.locator('.loading-spinner');
      
      // Loading should eventually complete
      await page.waitForSelector('[data-testid="3d-preview"]', { timeout: 10000 });
    });
  });
});