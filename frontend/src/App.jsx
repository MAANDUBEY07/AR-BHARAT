import React, { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { motion } from "framer-motion";
import Home from "./pages/Home";
import UploadPage from "./pages/UploadPage";
import GalleryPage from "./pages/GalleryPage";
import ARPage from "./pages/ARPage";
import BlogPage from "./pages/BlogPage";
import ContactPage from "./pages/ContactPage";
import Header from "./components/Header";
import Chatbot from "./components/Chatbot";
import ChatbotButton from "./components/ChatbotButton";

export default function App(){
  const [chatbotOpen, setChatbotOpen] = useState(false);
  const [currentPatternMetadata, setCurrentPatternMetadata] = useState(null);

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <Header />
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
          className="max-w-6xl mx-auto p-6"
        >
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={<UploadPage setPatternMetadata={setCurrentPatternMetadata} />} />
            <Route path="/gallery" element={<GalleryPage />} />
            <Route path="/ar/:id" element={<ARPage />} />
            <Route path="/ar" element={<ARPage />} />
            <Route path="/blog" element={<BlogPage />} />
            <Route path="/contact" element={<ContactPage />} />
          </Routes>
        </motion.div>
        
        {/* Chatbot */}
        <ChatbotButton 
          onClick={() => setChatbotOpen(true)} 
          isOpen={chatbotOpen} 
        />
        <Chatbot 
          isOpen={chatbotOpen} 
          onClose={() => setChatbotOpen(false)}
          patternMetadata={currentPatternMetadata}
        />
      </div>
    </BrowserRouter>
  );
}