import React from 'react';
import { motion } from 'framer-motion';

export default function ChatbotButton({ onClick, isOpen }) {
  if (isOpen) return null;

  return (
    <motion.button
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      className="fixed right-6 bottom-6 w-14 h-14 bg-gradient-to-r from-orange-500 via-green-500 to-blue-500 text-white rounded-full shadow-2xl hover:shadow-3xl transition-shadow z-40 flex items-center justify-center text-2xl"
    >
      🤖
    </motion.button>
  );
}