import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

export default function BlogPage() {
  const navigate = useNavigate();
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [aiGenerated, setAiGenerated] = useState(false);
  const [contentSource, setContentSource] = useState('Curated Content');

  // Fetch articles from OpenAI-powered backend
  const fetchArticles = async () => {
    try {
      setLoading(true);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000); // 8 second timeout
      
      const response = await fetch('/api/blog/articles?featured=1&limit=6', {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error('Failed to fetch articles');
      }
      const data = await response.json();
      setArticles(data.articles || []);
      setLastUpdated(data.generated_at);
      setAiGenerated(data.ai_generated || false);
      setContentSource(data.source || 'Curated Content');
      
      // Only show success message if AI content was actually generated
      if (!data.ai_generated) {
        setError('AI generation timed out. Showing curated content.');
      } else {
        setError(null);
      }
    } catch (err) {
      console.error('Error fetching articles:', err);
      setError('AI generation timed out. Showing fallback content.');
      // Fallback to fresh static content with relevant imagery
      const currentDate = new Date().toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
      setArticles([
        { 
          id: `client_fallback_${Date.now()}_1`,
          title: "Virtual Kolam: Sacred Geometry in Digital Space", 
          date: currentDate, 
          excerpt: "Sacred geometry meets digital innovation as AR technology transforms traditional Kolam art for global accessibility.",
          imageUrl: "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=800&h=400&fit=crop&auto=format&q=80",
          featured: true,
          category: "Cultural Technology",
          author: "AR BHARAT Team",
          readTime: "6 min read"
        },
        { 
          id: `client_fallback_${Date.now()}_2`,
          title: "Rangoli Patterns: Ancient Mathematics Revealed", 
          date: currentDate, 
          excerpt: "Ancient Rangoli designs reveal sophisticated mathematical principles that continue to inspire modern geometric art.",
          imageUrl: "https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?w=400&h=250&fit=crop&auto=format&q=80",
          featured: false,
          category: "Mathematical Heritage",
          author: "AR BHARAT Cultural Team", 
          readTime: "5 min read"
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Auto-refresh content every 30 minutes
  useEffect(() => {
    fetchArticles();
    
    const refreshInterval = setInterval(() => {
      fetchArticles();
    }, 30 * 60 * 1000); // 30 minutes

    return () => clearInterval(refreshInterval);
  }, []);

  // Manual refresh function
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      // Trigger backend refresh
      const refreshResponse = await fetch('/api/blog/refresh', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ type: 'all' })
      });
      
      if (refreshResponse.ok) {
        // Fetch updated articles
        await fetchArticles();
      }
    } catch (err) {
      console.error('Error refreshing content:', err);
    } finally {
      setRefreshing(false);
    }
  };

  const featuredPost = articles.find(article => article.featured) || articles[0];
  const regularPosts = articles.filter(article => !article.featured).slice(0, 5);

  if (loading && articles.length === 0) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Generating fresh content with AI...</p>
        </div>
      </div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="max-w-6xl mx-auto"
      data-testid="blog-page"
    >
      {/* Hero Section */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="text-center mb-12"
      >
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800">
            Heritage Chronicles
          </h1>
          <div className="flex items-center gap-4">
            {/* Real-time indicator */}
            <div className="flex items-center gap-2 text-sm text-green-600" data-testid="live-indicator">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Live AI Content</span>
            </div>
            {/* Manual refresh button */}
            <motion.button
              onClick={handleRefresh}
              disabled={refreshing}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-purple-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-purple-700 transition-colors disabled:opacity-50"
              data-testid="refresh-blog-button"
            >
              {refreshing ? (
                <div className="flex items-center gap-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                  Refreshing...
                </div>
              ) : (
                '🔄 Refresh Content'
              )}
            </motion.button>
          </div>
        </div>
        
        <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-4">
          Real-time AI-powered exploration of technology, culture, and heritage preservation. 
          Discover how digital innovation is revolutionizing our connection to traditional art forms.
        </p>
        
        {/* Status indicators */}
        <div className="flex justify-center gap-6 text-sm text-gray-500 mb-2">
          <span data-testid="openai-branding">
            {aiGenerated ? '🤖 AI Generated' : '📚 Curated Content'} • Powered by OpenAI
          </span>
          <span data-testid="article-count">📝 {articles.length} Articles</span>
          {lastUpdated && (
            <span data-testid="last-updated">🕐 Updated {new Date(lastUpdated).toLocaleTimeString()}</span>
          )}
        </div>
        
        {error && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 max-w-md mx-auto">
            <p className="text-yellow-800 text-sm">⚠️ {error}</p>
          </div>
        )}
      </motion.div>

      {/* Featured Post */}
      {featuredPost && (
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="relative rounded-2xl overflow-hidden mb-12 text-white"
          data-testid="featured-article"
        >
          {/* Background image */}
          <div className="absolute inset-0 min-h-[320px]">
            <img 
              src={featuredPost.imageUrl || "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=800&h=400&fit=crop&auto=format&q=80"}
              alt={featuredPost.title}
              className="w-full h-full object-cover"
              onError={(e) => {
                console.error('Featured image failed to load:', e.target.src);
                e.target.style.display = 'none';
              }}
            />
            <div className="absolute inset-0 bg-gradient-to-r from-purple-900/90 to-indigo-900/90"></div>
          </div>
          <div className="relative p-8 md:p-12 min-h-[320px]">
            <div className="flex items-center gap-4 mb-4">
              <span className="bg-white/20 px-3 py-1 rounded-full text-sm font-medium">Featured</span>
              <span className="text-sm opacity-90">{featuredPost.date}</span>
              {featuredPost.category && (
                <span className="bg-white/10 px-2 py-1 rounded text-xs">{featuredPost.category}</span>
              )}
              {featuredPost.ai_unavailable && (
                <span className="bg-yellow-500/20 px-2 py-1 rounded text-xs">⚡ Fallback Content</span>
              )}
            </div>
            <h2 className="text-3xl md:text-4xl font-bold mb-4">{featuredPost.title}</h2>
            <p className="text-lg md:text-xl opacity-90 mb-6 max-w-3xl">{featuredPost.excerpt}</p>
            
            {/* Article metadata */}
            <div className="flex items-center gap-4 mb-6 text-sm opacity-75">
              {featuredPost.author && <span data-testid="article-author">👤 {featuredPost.author}</span>}
              {featuredPost.readTime && <span data-testid="article-readtime">⏱️ {featuredPost.readTime}</span>}
            </div>
            
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-white text-purple-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              data-testid="read-full-article"
              onClick={() => {
                navigate(`/blog/article/${featuredPost.id}`);
              }}
            >
              Read Full Article →
            </motion.button>
          </div>
        </motion.div>
      )}

      {/* Blog Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8" data-testid="articles-grid">
        {regularPosts.map((post, i) => (
          <motion.article 
            key={post.id || i}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 * (i + 1) }}
            whileHover={{ y: -5 }}
            className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-2xl transition-all duration-300"
            data-testid={`article-card-${i}`}
          >
            <div className="h-48 relative overflow-hidden">
              <img 
                src={post.imageUrl || "https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?w=400&h=250&fit=crop&auto=format&q=80"}
                alt={post.title}
                className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
              />
              {post.ai_unavailable && (
                <div className="absolute top-2 right-2 bg-yellow-500 text-white text-xs px-2 py-1 rounded">
                  Fallback
                </div>
              )}
            </div>
            <div className="p-6">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-indigo-600 font-medium">{post.date}</div>
                {post.category && (
                  <div className="text-xs bg-gray-100 px-2 py-1 rounded text-gray-600">
                    {post.category}
                  </div>
                )}
              </div>
              <h3 className="text-xl font-bold text-gray-800 mb-3 line-clamp-2">{post.title}</h3>
              <p className="text-gray-600 mb-4 line-clamp-3">{post.excerpt}</p>
              
              {/* Article metadata */}
              <div className="flex items-center justify-between mb-4 text-xs text-gray-500">
                <span>{post.author || "AR BHARAT Team"}</span>
                <span>{post.readTime || "3 min read"}</span>
              </div>
              
              <motion.div 
                whileHover={{ x: 5 }}
                className="text-indigo-600 font-semibold text-sm cursor-pointer"
                onClick={() => {
                  navigate(`/blog/article/${post.id}`);
                }}
                data-testid="read-more-link"
              >
                Read more →
              </motion.div>
            </div>
          </motion.article>
        ))}
        
        {/* Show placeholder if no articles */}
        {regularPosts.length === 0 && !loading && (
          <div className="col-span-full text-center py-12">
            <div className="text-gray-400 mb-4">
              <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
              </svg>
            </div>
            <p className="text-gray-600">No articles available. Try refreshing to generate new content.</p>
          </div>
        )}
      </div>

      {/* Newsletter Section */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 1 }}
        className="bg-gray-50 rounded-2xl p-8 md:p-12 mt-16 text-center"
        data-testid="newsletter-section"
      >
        <h3 className="text-2xl md:text-3xl font-bold text-gray-800 mb-4">
          Stay Updated with AI-Powered Heritage Tech
        </h3>
        <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
          Get real-time updates on digital heritage preservation, AI in cultural studies, and innovative AR experiences. 
          Our content is powered by OpenAI for the latest insights.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto mb-6">
          <input 
            type="email" 
            placeholder="Enter your email" 
            className="flex-1 px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            data-testid="email-input"
          />
          <motion.button 
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition-colors"
            data-testid="subscribe-button"
          >
            Subscribe
          </motion.button>
        </div>
        
        {/* AI features callout */}
        <div className="bg-white rounded-lg p-6 max-w-xl mx-auto" data-testid="ai-features">
          <h4 className="font-bold text-gray-800 mb-2">🤖 AI-Enhanced Features</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600">
            <div>
              <span className="font-medium">Real-time Content</span>
              <br />Fresh articles generated with OpenAI
            </div>
            <div>
              <span className="font-medium">Trending Topics</span>
              <br />AI-curated heritage insights
            </div>
            <div>
              <span className="font-medium">Auto Updates</span>
              <br />Content refreshed every 5 minutes
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}