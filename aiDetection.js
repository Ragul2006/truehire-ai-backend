/**
 * Local Heuristic AI Detection
 * Returns whether a given string of text is likely AI generated based on heuristics.
 * Rules implemented:
 * 1. Sentence length variance: Human speech tends to have high variance, AI can be too uniform.
 * 2. Repetition patterns: Humans repeat words/fillers more frequently in raw speech.
 * 3. Predictability: Certain AI model transitional phrases ("In conclusion", "It is important to note").
 * 4. Grammar uniformity: Strict vs loose grammar structure (human speech is often fragmented).
 */

const AI_PHRASES = [
    "in conclusion", "it is important to note", "moreover", "furthermore",
    "as an ai", "i cannot", "additionally", "nevertheless", "to summarize",
    "can be defined as", "plays a crucial role", "firstly", "secondly"
];

function calculateVariance(array) {
    if (array.length === 0) return 0;
    const mean = array.reduce((a, b) => a + b) / array.length;
    const sqDiffs = array.map(value => Math.pow(value - mean, 2));
    const avgSqDiff = sqDiffs.reduce((a, b) => a + b) / array.length;
    return avgSqDiff;
}

function detectAIGeneratedText(text) {
    if (!text || text.trim().length === 0) {
        return { aiGenerated: false, confidence: 0, details: "Empty text" };
    }

    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const words = text.toLowerCase().match(/\b\w+\b/g) || [];

    let score = 0; // Higher score = more likely AI
    let maxScore = 4;
    
    // 1. Sentence length variance
    const sentenceLengths = sentences.map(s => s.split(' ').length);
    const lengthVariance = calculateVariance(sentenceLengths);
    // Humans have mixed short and long sentences. Low variance might indicate AI.
    if (sentenceLengths.length > 2 && lengthVariance < 10) {
        score += 1;
    }

    // 2. Predictability (AI specific phrases)
    const lowerText = text.toLowerCase();
    let phraseMatches = 0;
    for (const phrase of AI_PHRASES) {
        if (lowerText.includes(phrase)) {
            phraseMatches += 1;
        }
    }
    if (phraseMatches >= 1) {
        score += (phraseMatches > 2 ? 1.5 : 1);
    }

    // 3. Repetition Patterns (Vocabulary richness / Lexical Diversity)
    // Humans have more filler words and repeating patterns in small chunks,
    // AI has high vocabulary richness but structural repetition.
    const uniqueWords = new Set(words).size;
    const lexicalDiversity = words.length > 0 ? (uniqueWords / words.length) : 0;
    // Overly high lexical diversity in normal speech is somewhat robotic/AI
    if (words.length > 20 && lexicalDiversity > 0.75) {
        score += 0.5;
    }

    // 4. Grammar Uniformity (Proxy: checking for typical human filler words vs AI perfection)
    const humanFillers = ['uh', 'um', 'like', 'you know', 'sort of', 'kinda', 'basically'];
    let fillerCount = 0;
    for (const filler of humanFillers) {
        const regex = new RegExp(`\\b${filler}\\b`, 'g');
        const matches = lowerText.match(regex);
        if (matches) fillerCount += matches.length;
    }
    // Lack of filler words in speech transcription indicates high structure
    if (fillerCount === 0 && words.length > 30) {
        score += 1;
    } else if (fillerCount > 2) {
        score -= 1; // Reduces AI probability
    }

    // Normalize confidence
    let confidence = Math.max(0, Math.min(1, score / maxScore));
    
    // Threshold
    let aiGenerated = confidence > 0.6;

    return {
        aiGenerated: aiGenerated,
        confidence: parseFloat(confidence.toFixed(2)),
        details: {
            score,
            variance: lengthVariance,
            lexicalDiversity,
            fillerCount,
            phraseMatches
        }
    };
}

module.exports = { detectAIGeneratedText };
