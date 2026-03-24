const vosk = require('vosk');
const fs = require('fs');

const MODEL_PATH = "model";
let model;
let initialized = false;

function initVosk() {
    if (initialized) return true;
    if (!fs.existsSync(MODEL_PATH)) {
        console.error(`Please download the model from https://alphacephei.com/vosk/models and unpack as ${MODEL_PATH} in the server folder.`);
        return false;
    }
    vosk.setLogLevel(-1); // Disable verbose vosk logs
    model = new vosk.Model(MODEL_PATH);
    initialized = true;
    return true;
}

/**
 * Transcribes a raw 16kHz PCM audio buffer using Vosk.
 * @param {Buffer} audioBuffer - the raw PCM audio bytes.
 * @returns {Promise<string>}
 */
function transcribeAudio(audioBuffer) {
    return new Promise((resolve, reject) => {
        try {
            if (!initVosk()) {
                return reject(new Error("Vosk model not found. Cannot transcribe."));
            }
            
            // Expected sample rate from the frontend's audio buffer
            const rec = new vosk.Recognizer({ model: model, sampleRate: 16000 });
            
            // For streams or chunks, acceptWaveform processes the audio
            let transcribedText = '';
            
            // acceptWaveform returns true if silence is detected and a chunk is ready
            if (rec.acceptWaveform(audioBuffer)) {
                transcribedText = rec.result().text;
            } else {
                transcribedText = rec.partialResult().partial;
            }
            
            // Also append final result to be safe
            const finalResult = rec.finalResult();
            if (finalResult.text && !transcribedText.includes(finalResult.text)) {
                transcribedText += " " + finalResult.text;
            }

            rec.free();
            resolve(transcribedText.trim());
        } catch (error) {
            console.error('Failed to transcribe audio:', error);
            reject(error);
        }
    });
}

module.exports = { transcribeAudio };
