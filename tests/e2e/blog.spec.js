import { test, expect } from '@playwright/test';

test.describe('Real-Time Blog Integration', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the blog page before each test
    await page.goto('http://localhost:5173/blog');
  });

  test('should load blog page with AI content indicator', async ({ page }) => {
    // Check that the page loads with the correct title
    await expect(page).toHaveTitle('AR BHARAT');
    
    // Verify blog page structure is present
    await expect(page.locator('h1')).toContainText('Heritage Chronicles');
    
    // Check for "Live AI Content" indicator
    await expect(page.locator('[data-testid="live-indicator"]')).toContainText('Live AI Content');
    
    // Verify OpenAI branding is visible
    await expect(page.locator('[data-testid="openai-branding"]')).toContainText('🤖 Powered by OpenAI');
  });

  test('should display articles with proper structure', async ({ page }) => {
    // Wait for content to load (should be instant now)
    await page.waitForLoadState('networkidle');
    
    // Check that articles are displayed
    await expect(page.locator('article')).toHaveCount(5); // 5 regular articles
    
    // Verify featured article exists
    const featuredArticle = page.locator('[data-testid="featured-article"]');
    await expect(featuredArticle).toBeVisible();
    await expect(featuredArticle.locator('h2')).toContainText('AI-Powered Heritage Preservation');
    
    // Check article metadata
    await expect(page.locator('[data-testid="article-author"]').first()).toContainText('AR BHARAT');
    await expect(page.locator('[data-testid="article-readtime"]').first()).toContainText('min read');
  });

  test('should show real-time metadata', async ({ page }) => {
    // Verify article count display
    await expect(page.locator('[data-testid="article-count"]')).toContainText('📝 6 Articles');
    
    // Check last updated timestamp is shown
    const timestampElement = page.locator('[data-testid="last-updated"]');
    await expect(timestampElement).toBeVisible();
    await expect(timestampElement).toContainText('🕐 Updated');
  });

  test('should handle manual refresh functionality', async ({ page }) => {
    // Find the refresh button
    const refreshButton = page.locator('[data-testid="refresh-blog-button"]');
    await expect(refreshButton).toBeVisible();
    await expect(refreshButton).toContainText('🔄 Refresh Content');
    
    // Click refresh button
    await refreshButton.click();
    
    // Verify loading state
    await expect(refreshButton).toContainText('Refreshing...');
    await expect(refreshButton).toBeDisabled();
    
    // Wait for refresh to complete (should be quick with fallback content)
    await page.waitForTimeout(3000);
    
    // Verify button returns to normal state
    await expect(refreshButton).not.toBeDisabled();
    await expect(refreshButton).toContainText('🔄 Refresh Content');
  });

  test('should display AI features section', async ({ page }) => {
    // Check AI features callout section
    const aiSection = page.locator('[data-testid="ai-features"]');
    await expect(aiSection).toBeVisible();
    await expect(aiSection.locator('h4')).toContainText('🤖 AI-Enhanced Features');
    
    // Verify feature list items
    await expect(aiSection.locator('text=Real-time Content')).toBeVisible();
    await expect(aiSection.locator('text=Fresh articles generated with OpenAI')).toBeVisible();
    await expect(aiSection.locator('text=Trending Topics')).toBeVisible();
    await expect(aiSection.locator('text=AI-curated heritage insights')).toBeVisible();
    await expect(aiSection.locator('text=Auto Updates')).toBeVisible();
    await expect(aiSection.locator('text=Content refreshed every 5 minutes')).toBeVisible();
  });

  test('should have functional article links', async ({ page }) => {
    // Test "Read Full Article" button on featured article
    const readFullButton = page.locator('[data-testid="read-full-article"]');
    await expect(readFullButton).toBeVisible();
    await expect(readFullButton).toContainText('Read Full Article →');
    
    // Test "Read more" links on regular articles
    const readMoreLinks = page.locator('[data-testid="read-more-link"]');
    await expect(readMoreLinks.first()).toBeVisible();
    await expect(readMoreLinks.first()).toContainText('Read more →');
  });

  test('should display proper categorization', async ({ page }) => {
    // Check that articles have categories
    const categories = [
      'Technology',
      'Cultural Heritage', 
      'Global Heritage',
      'Education',
      'Cultural Evolution',
      'Artificial Intelligence'
    ];
    
    for (const category of categories) {
      await expect(page.locator(`text=${category}`).first()).toBeVisible();
    }
  });

  test('should handle newsletter subscription section', async ({ page }) => {
    // Check newsletter section exists
    const newsletterSection = page.locator('[data-testid="newsletter-section"]');
    await expect(newsletterSection).toBeVisible();
    await expect(newsletterSection.locator('h3')).toContainText('Stay Updated with AI-Powered Heritage Tech');
    
    // Verify email input and subscribe button
    const emailInput = page.locator('[data-testid="email-input"]');
    const subscribeButton = page.locator('[data-testid="subscribe-button"]');
    
    await expect(emailInput).toBeVisible();
    await expect(emailInput).toHaveAttribute('placeholder', 'Enter your email');
    await expect(subscribeButton).toBeVisible();
    await expect(subscribeButton).toContainText('Subscribe');
  });

  test('should be responsive and accessible', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.reload();
    
    // Verify main elements are still visible in mobile
    await expect(page.locator('h1')).toContainText('Heritage Chronicles');
    await expect(page.locator('[data-testid="live-indicator"]')).toBeVisible();
    await expect(page.locator('[data-testid="refresh-blog-button"]')).toBeVisible();
    
    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.reload();
    
    // Verify layout adapts properly
    await expect(page.locator('[data-testid="featured-article"]')).toBeVisible();
    await expect(page.locator('article').first()).toBeVisible();
  });

  test('should handle chatbot integration', async ({ page }) => {
    // Verify chatbot button is present
    const chatbotButton = page.locator('button').filter({ hasText: '🤖' }).last();
    await expect(chatbotButton).toBeVisible();
    
    // Click chatbot button to test integration
    await chatbotButton.click();
    
    // Note: Actual chatbot functionality would be tested separately
    // This just ensures the button is present and clickable
  });

  test('should show proper error handling', async ({ page }) => {
    // Test with network offline to trigger fallback content
    await page.context().setOffline(true);
    await page.reload();
    
    // Should still show content (fallback articles)
    await expect(page.locator('h1')).toContainText('Heritage Chronicles');
    
    // Re-enable network
    await page.context().setOffline(false);
  });
});

test.describe('Blog API Integration', () => {
  test('should fetch articles from backend API', async ({ page }) => {
    // Navigate to blog and monitor network requests
    await page.goto('http://localhost:5173/blog');
    
    // Wait for API call
    const apiResponse = await page.waitForResponse('**/api/blog/articles*');
    expect(apiResponse.status()).toBe(200);
    
    const responseBody = await apiResponse.json();
    expect(responseBody.articles).toBeDefined();
    expect(responseBody.articles.length).toBeGreaterThan(0);
  });

  test('should handle API timeout gracefully', async ({ page }) => {
    // Mock slow API response to test timeout handling
    await page.route('**/api/blog/articles*', async route => {
      await new Promise(resolve => setTimeout(resolve, 1000)); // 1 second delay
      await route.continue();
    });
    
    await page.goto('http://localhost:5173/blog');
    
    // Should still load with fallback content
    await expect(page.locator('h1')).toContainText('Heritage Chronicles');
    await expect(page.locator('[data-testid="featured-article"]')).toBeVisible();
  });
});