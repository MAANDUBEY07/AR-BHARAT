from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Pattern(db.Model):
    __tablename__ = 'patterns'
    
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, nullable=False)  # UUID for external reference
    name = db.Column(db.String(140), default='untitled')
    filename_png = db.Column(db.String(300))
    filename_svg = db.Column(db.String(300))
    svg_content = db.Column(db.Text)  # Store SVG content directly
    meta = db.Column(db.Text)  # JSON metadata
    artist = db.Column(db.String(100), default='Anonymous')
    likes = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert pattern to dictionary for JSON responses"""
        return {
            'id': self.uuid,  # Use UUID for external API
            'svg': self.svg_content,
            'metadata': json.loads(self.meta) if self.meta else {},
            'artist': self.artist,
            'likes': self.likes,
            'created': self.created_at.isoformat() + 'Z',
            'name': self.name,
            'filename_png': self.filename_png,
            'filename_svg': self.filename_svg
        }
    
    def set_metadata(self, metadata_dict):
        """Set metadata from dictionary"""
        self.meta = json.dumps(metadata_dict) if metadata_dict else None
    
    def get_metadata(self):
        """Get metadata as dictionary"""
        return json.loads(self.meta) if self.meta else {}