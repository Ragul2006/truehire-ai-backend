const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { transcribeAudio } = require('./speechService');
const { detectAIGeneratedText } = require('./aiDetection');

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// Set up memory storage for uploaded audio blobs
const upload = multer({ storage: multer.memoryStorage() });

// Endpoint to process uploaded audio and return AI detection
app.post('/speech-detect', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No audio file uploaded.' });
        }
        
        console.log('Received audio blob, size:', req.file.size);
        
        // Pass the audio buffer to Vosk
        const text = await transcribeAudio(req.file.buffer);
        
        if (!text || text.trim() === '') {
            return res.json({
                text: '',
                aiGenerated: false,
                confidence: 0,
                details: { info: "No speech or silent audio" }
            });
        }
        
        console.log(`Recognized Text: ${text}`);

        // Run local heuristic AI detection
        const detectionResult = detectAIGeneratedText(text);

        res.json({
            text: text,
            aiGenerated: detectionResult.aiGenerated,
            confidence: detectionResult.confidence,
            details: detectionResult.details
        });
    } catch (error) {
        console.error('Error in /speech-detect:', error);
        res.status(500).json({ error: 'Internal server error processing speech detection.' });
    }
});

app.listen(PORT, () => {
    console.log(`TrueHire Backend is running on http://localhost:${PORT}`);
    console.log('Ensure you have the Vosk model downloaded in the "model" folder.');
});
