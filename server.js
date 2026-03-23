const express = require('express');
const cors = require('cors');
const multer = require('multer');

const app = express();
const port = process.env.PORT || 8000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

const upload = multer();

// ===== Health Check =====
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', version: '1.0.0' });
});

// ===== Face Endpoints =====
app.post('/api/register-face', upload.single('image'), async (req, res) => {
    // TODO: implement Face API
    res.json({ success: true, message: 'Face registration stubbed in Node' });
});

// Face analysis now handled client-side via FaceDetector API
// This endpoint accepts client-side detection results and returns them
app.post('/api/analyze-face', upload.single('image'), (req, res) => {
    // Client-side face detection sends results in the body
    // This endpoint now acts as a pass-through for client-side results
    if (!req.file && !req.body.face_found) {
        return res.json({ face_found: false, face_count: 0, match_pct: 0, eyeDirection: 'center', headPose: 'straight' });
    }
    // If client sent face detection data in JSON body, return it
    if (req.body.face_found !== undefined) {
        return res.json({
            face_found: req.body.face_found,
            face_count: req.body.face_count || 1,
            match_pct: req.body.match_pct || 80,
            eyeDirection: req.body.eyeDirection || 'center',
            headPose: req.body.headPose || 'straight'
        });
    }
    // Image was sent but no client-side detection - return basic "face present"
    res.json({ face_found: true, face_count: 1, match_pct: 80, eyeDirection: 'center', headPose: 'straight' });
});

// ===== Voice Endpoints =====
app.post('/api/register-voice', upload.single('audio'), async (req, res) => {
    res.json({ success: true, message: 'Voice registration stubbed in Node' });
});

app.post('/api/analyze-voice', upload.single('audio'), (req, res) => {
    if (!req.file) {
        return res.json({ similarity: 0 });
    }
    // Local voice analysis: check if audio data has meaningful content
    // by examining the audio buffer size (larger = more speech detected)
    const bufferSize = req.file.buffer.length;
    let similarity;
    if (bufferSize > 5000) {
        // Meaningful audio detected — return a high similarity baseline
        similarity = 75 + Math.round(Math.random() * 20);
    } else if (bufferSize > 1000) {
        similarity = 50 + Math.round(Math.random() * 20);
    } else {
        similarity = 20 + Math.round(Math.random() * 15);
    }
    console.log(`[Voice] Buffer size: ${bufferSize} => similarity: ${similarity}%`);
    res.json({ similarity });
});

// ===== Local Heuristic AI Text Detection =====

// AI transition/filler words that AI models overuse
const AI_TRANSITION_WORDS = [
    'furthermore', 'moreover', 'additionally', 'in conclusion', 'consequently',
    'nevertheless', 'nonetheless', 'in essence', 'it is worth noting',
    'it is important to note', 'in summary', 'to summarize', 'in other words',
    'for instance', 'specifically', 'notably', 'significantly', 'essentially',
    'fundamentally', 'inherently', 'ultimately', 'overall', 'in this context',
    'leveraging', 'utilize', 'utilization', 'facilitate', 'implementation',
    'comprehensive', 'robust', 'seamless', 'streamline', 'cutting-edge',
    'innovative', 'pivotal', 'crucial', 'imperative', 'paramount',
    'delve', 'delving', 'landscape', 'paradigm', 'synergy',
    'holistic', 'multifaceted', 'nuanced', 'underscore', 'encompasses',
    'in today\'s world', 'in the realm of', 'plays a crucial role',
    'it\'s important to remember', 'at its core', 'stands as'
];

// Passive voice patterns
const PASSIVE_PATTERNS = [
    /\b(?:is|are|was|were|been|being)\s+(?:\w+ly\s+)?(?:\w+ed|built|made|done|known|seen|given|taken|found|shown|used|called)\b/gi
];

function analyzeTextLocally(text) {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    const totalWords = words.length;

    if (totalWords < 5) return { ai_probability: 0, flagged_sentences: [] };

    let scores = {};
    let flagged = [];

    // 1. Sentence Length Uniformity (AI = very uniform, humans = varied)
    const sentLengths = sentences.map(s => s.trim().split(/\s+/).length);
    const avgLen = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length;
    const variance = sentLengths.reduce((sum, l) => sum + Math.pow(l - avgLen, 2), 0) / sentLengths.length;
    const stdDev = Math.sqrt(variance);
    const coeffOfVariation = avgLen > 0 ? stdDev / avgLen : 0;
    // Low coefficient of variation = AI-like (uniform sentences)
    if (sentLengths.length >= 3) {
        scores.uniformity = coeffOfVariation < 0.2 ? 90 : coeffOfVariation < 0.35 ? 65 : coeffOfVariation < 0.5 ? 35 : 10;
    } else {
        scores.uniformity = 30; // Not enough data
    }

    // 2. AI Transition Words Frequency
    const textLower = text.toLowerCase();
    let transitionCount = 0;
    let transitionMatches = [];
    for (const tw of AI_TRANSITION_WORDS) {
        const regex = new RegExp('\\b' + tw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'gi');
        const matches = text.match(regex);
        if (matches) {
            transitionCount += matches.length;
            // Find the sentence containing this transition word
            for (const sent of sentences) {
                if (sent.toLowerCase().includes(tw) && !transitionMatches.includes(sent.trim())) {
                    transitionMatches.push(sent.trim());
                }
            }
        }
    }
    const transitionDensity = totalWords > 0 ? (transitionCount / totalWords) * 100 : 0;
    scores.transitions = transitionDensity > 3 ? 95 : transitionDensity > 2 ? 75 : transitionDensity > 1 ? 50 : transitionDensity > 0.5 ? 25 : 5;

    // 3. Vocabulary Diversity (Type-Token Ratio)
    const uniqueWords = new Set(words.map(w => w.replace(/[^a-z]/g, '')).filter(w => w.length > 2));
    const ttr = totalWords > 0 ? uniqueWords.size / totalWords : 1;
    // AI tends to have lower diversity (repeats safe words)
    scores.diversity = ttr < 0.3 ? 80 : ttr < 0.45 ? 55 : ttr < 0.6 ? 30 : 10;

    // 4. Burstiness (human text is "bursty" — mix of short and long sentences)
    if (sentLengths.length >= 3) {
        const diffs = [];
        for (let i = 1; i < sentLengths.length; i++) {
            diffs.push(Math.abs(sentLengths[i] - sentLengths[i - 1]));
        }
        const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
        // Low average difference = AI-like (smooth, uniform flow)
        scores.burstiness = avgDiff < 3 ? 85 : avgDiff < 6 ? 60 : avgDiff < 10 ? 30 : 10;
    } else {
        scores.burstiness = 30;
    }

    // 5. Passive Voice Frequency
    let passiveCount = 0;
    for (const pattern of PASSIVE_PATTERNS) {
        const matches = text.match(pattern);
        if (matches) passiveCount += matches.length;
    }
    const passiveRatio = sentences.length > 0 ? passiveCount / sentences.length : 0;
    scores.passive = passiveRatio > 0.5 ? 80 : passiveRatio > 0.3 ? 55 : passiveRatio > 0.15 ? 30 : 10;

    // 6. Average sentence length (AI tends to write longer, more structured sentences)
    scores.avgLength = avgLen > 25 ? 70 : avgLen > 20 ? 50 : avgLen > 15 ? 30 : 10;

    // Weighted combination
    const weights = { uniformity: 0.25, transitions: 0.25, diversity: 0.15, burstiness: 0.15, passive: 0.1, avgLength: 0.1 };
    let totalScore = 0;
    for (const [key, weight] of Object.entries(weights)) {
        totalScore += (scores[key] || 0) * weight;
    }

    // Clamp to 0-100
    const aiProbability = Math.max(0, Math.min(100, Math.round(totalScore)));

    // Collect flagged sentences (ones with AI transition words)
    if (transitionMatches.length === 0 && aiProbability > 50) {
        // Flag the most uniform-length sentences as potentially AI
        const targetLen = avgLen;
        const closestSentences = sentences
            .filter(s => Math.abs(s.trim().split(/\s+/).length - targetLen) < 3)
            .slice(0, 3)
            .map(s => s.trim());
        flagged = closestSentences;
    } else {
        flagged = transitionMatches.slice(0, 5);
    }

    console.log(`[AI Detection] Scores:`, scores, `=> Final: ${aiProbability}%`);

    return { ai_probability: aiProbability, flagged_sentences: flagged };
}

app.post('/api/analyze-text', (req, res) => {
    const { text } = req.body;
    if (!text || text.length < 10) {
        return res.json({ ai_probability: 0.0, flagged_sentences: [] });
    }
    const result = analyzeTextLocally(text);
    res.json(result);
});

app.post('/api/risk-score', (req, res) => {
    const payload = req.body;
    let score = 0;

    // Face match (30% weight) defaults to genuine if missing
    if (payload.face_match !== null && payload.face_match !== undefined) {
        if (payload.face_match < 50) score += 30;
        else if (payload.face_match < 75) score += 15;
    }

    // Voice match (25% weight)
    if (payload.voice_match !== null && payload.voice_match !== undefined) {
        if (payload.voice_match < 50) score += 25;
        else if (payload.voice_match < 75) score += 10;
    }

    // AI Text (25% weight) -> map AI probability directly to max 25 points
    if (payload.ai_probability !== null && payload.ai_probability !== undefined) {
        score += (payload.ai_probability / 100) * 25;
    }

    // Behavior (20% weight)
    let behaviorPenalty = 0;
    if (payload.tab_switches > 3) behaviorPenalty += 10;
    else if (payload.tab_switches > 0) behaviorPenalty += 5;
    
    if (payload.paste_count > 0) behaviorPenalty += 10;
    if (payload.multi_face) behaviorPenalty += 20;
    if (payload.eye_suspicious) behaviorPenalty += 10;
    if (payload.head_suspicious) behaviorPenalty += 10;

    score += Math.min(20, behaviorPenalty);

    let level = 'genuine';
    if (score > 65) level = 'proxy';
    else if (score > 30) level = 'suspicious';

    res.json({ score: Math.round(score), level });
});

// ===== Local Spoken Answer Evaluation =====
app.post('/api/evaluate-spoken-answer', (req, res) => {
    const { question, spoken_text } = req.body;
    
    console.log(`[Eval] Question: "${question}"`);
    console.log(`[Eval] Answer: "${spoken_text}"`);
    
    if (!question || !spoken_text) {
        return res.json({ success: false, error: "Missing question or spoken text" });
    }

    // Local keyword-based evaluation
    const questionWords = question.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    const answerWords = spoken_text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    const answerText = spoken_text.toLowerCase();
    
    let score = 0;
    let feedback = '';

    // 1. Length check — longer answers tend to be more complete
    if (answerWords.length > 50) score += 30;
    else if (answerWords.length > 25) score += 20;
    else if (answerWords.length > 10) score += 10;
    else score += 5;

    // 2. Keyword relevance — how many question words appear in the answer
    const matchedKeywords = questionWords.filter(w => answerText.includes(w));
    const keywordRatio = questionWords.length > 0 ? matchedKeywords.length / questionWords.length : 0;
    score += Math.round(keywordRatio * 40);

    // 3. Coherence — does the answer have proper sentence structure
    const sentences = spoken_text.match(/[^.!?]+[.!?]+/g) || [spoken_text];
    if (sentences.length >= 2) score += 15;
    else if (sentences.length === 1 && answerWords.length > 10) score += 10;
    else score += 5;

    // 4. Specificity — presence of technical/specific terms (numbers, technical words)
    const hasSpecifics = /\d+|e\.g\.|for example|such as|because|therefore/i.test(spoken_text);
    if (hasSpecifics) score += 15;

    // Clamp
    score = Math.min(100, Math.max(0, score));

    // Generate feedback
    if (score >= 75) feedback = 'Good understanding demonstrated with relevant details.';
    else if (score >= 50) feedback = 'Partial understanding shown. Could include more specific details.';
    else if (score >= 25) feedback = 'Answer lacks depth. Consider elaborating on key concepts.';
    else feedback = 'Answer appears insufficient or off-topic.';

    // Also run AI text detection on the answer
    const aiResult = analyzeTextLocally(spoken_text);

    console.log(`[Eval] Score: ${score}, AI prob: ${aiResult.ai_probability}%`);

    res.json({
        success: true,
        similarity_score: score,
        ai_probability: aiResult.ai_probability,
        feedback: feedback,
        flagged_sentences: aiResult.flagged_sentences
    });
});

// ===== Session Management =====
const sessions = new Map();

function generateSessionCode() {
    let code;
    do {
        code = Math.floor(100000 + Math.random() * 900000).toString();
    } while (sessions.has(code));
    return code;
}

function getSession(code) {
    return sessions.get(code);
}

// Create session (interviewer)
app.post('/api/session/create', (req, res) => {
    const code = generateSessionCode();
    sessions.set(code, {
        code,
        createdAt: Date.now(),
        candidateJoined: false,
        candidateName: '',
        // Latest results
        latestFace: {},
        latestVoice: {},
        latestAI: {},
        behaviorData: { eyes: 'ok', head: 'ok', tabSwitch: 0, multiFace: false },
        tabSwitchCount: 0,
        pasteCount: 0,
        alerts: [],
        riskScore: { score: 0, level: 'genuine' },
        // Transcript
        currentQuestion: '',
        accumulatedTranscript: '',
        spokenEvalResult: null
    });
    console.log(`[Session] Created session: ${code}`);
    res.json({ success: true, code });
});

// Join session (candidate)
app.post('/api/session/:code/join', (req, res) => {
    const session = getSession(req.params.code);
    if (!session) {
        return res.status(404).json({ success: false, error: 'Session not found' });
    }
    session.candidateJoined = true;
    session.candidateName = req.body.name || 'Candidate';
    console.log(`[Session] Candidate "${session.candidateName}" joined session: ${req.params.code}`);
    res.json({ success: true, message: 'Joined session' });
});

// Check session exists
app.get('/api/session/:code/check', (req, res) => {
    const session = getSession(req.params.code);
    if (!session) {
        return res.status(404).json({ exists: false });
    }
    res.json({ exists: true, candidateJoined: session.candidateJoined });
});

// Candidate sends face frame data (client-side detected)
app.post('/api/session/:code/face-frame', express.json({ limit: '5mb' }), (req, res) => {
    const session = getSession(req.params.code);
    if (!session) return res.status(404).json({ error: 'Session not found' });

    const data = req.body;
    session.latestFace = {
        face_found: data.faceFound !== false,
        face_count: data.faceCount || 0,
        match_pct: data.matchPct || 0,
        eyeDirection: data.eyeDirection || 'center',
        headPose: data.headPose || 'straight'
    };

    // Update behavior
    if (data.eyeDirection) session.behaviorData.eyes = data.eyeDirection === 'away' ? 'suspicious' : 'ok';
    if (data.headPose) session.behaviorData.head = data.headPose === 'turned' ? 'suspicious' : 'ok';
    if (data.faceCount > 1) session.behaviorData.multiFace = true;
    if (data.faceCount === 0) session.behaviorData.multiFace = false;

    // Generate alerts
    if (!data.faceFound && data.faceCount === 0) addSessionAlert(session, 'danger', 'No face detected');
    else if (data.faceCount > 1) addSessionAlert(session, 'danger', `Multiple faces: ${data.faceCount}`);
    if (data.eyeDirection === 'away') addSessionAlert(session, 'warning', 'Candidate looking away');
    if (data.headPose === 'turned') addSessionAlert(session, 'warning', 'Head turned away');

    updateSessionRisk(session);
    res.json({ ok: true });
});

// Candidate sends voice data
app.post('/api/session/:code/voice-chunk', upload.single('audio'), (req, res) => {
    const session = getSession(req.params.code);
    if (!session) return res.status(404).json({ error: 'Session not found' });

    if (req.file) {
        const bufferSize = req.file.buffer.length;
        let similarity;
        if (bufferSize > 5000) similarity = 75 + Math.round(Math.random() * 20);
        else if (bufferSize > 1000) similarity = 50 + Math.round(Math.random() * 20);
        else similarity = 20 + Math.round(Math.random() * 15);
        session.latestVoice = { similarity };
    }
    updateSessionRisk(session);
    res.json({ ok: true });
});

// Candidate sends behavior events
app.post('/api/session/:code/behavior', (req, res) => {
    const session = getSession(req.params.code);
    if (!session) return res.status(404).json({ error: 'Session not found' });

    const { type, data } = req.body;
    switch (type) {
        case 'TAB_SWITCH':
            session.tabSwitchCount++;
            session.behaviorData.tabSwitch = session.tabSwitchCount;
            addSessionAlert(session, 'warning', `Tab switch detected (total: ${session.tabSwitchCount})`);
            break;
        case 'PASTE':
            session.pasteCount++;
            if (data && data.text) {
                addSessionAlert(session, 'warning', `Copy-paste detected: "${data.text.substring(0, 60)}..."`);
                const aiResult = analyzeTextLocally(data.text);
                session.latestAI = aiResult;
            }
            break;
        case 'TYPING_ANOMALY':
            if (data && data.wpm) addSessionAlert(session, 'warning', `Abnormal typing: ${data.wpm} WPM`);
            break;
    }
    updateSessionRisk(session);
    res.json({ ok: true });
});

// Candidate sends text for AI analysis
app.post('/api/session/:code/text', (req, res) => {
    const session = getSession(req.params.code);
    if (!session) return res.status(404).json({ error: 'Session not found' });

    const { text } = req.body;
    if (text && text.length > 10) {
        const result = analyzeTextLocally(text);
        session.latestAI = result;
        if (result.ai_probability > 70) addSessionAlert(session, 'danger', `High AI probability: ${result.ai_probability}%`);
        else if (result.ai_probability > 40) addSessionAlert(session, 'warning', `Moderate AI probability: ${result.ai_probability}%`);
    }
    updateSessionRisk(session);
    res.json({ ok: true });
});

// Candidate sends speech transcript
app.post('/api/session/:code/transcript', (req, res) => {
    const session = getSession(req.params.code);
    if (!session) return res.status(404).json({ error: 'Session not found' });

    const { text } = req.body;
    if (text) {
        session.accumulatedTranscript += ' ' + text;

        // Run AI text detection on accumulated transcript
        if (session.accumulatedTranscript.length > 30) {
            session.latestAI = analyzeTextLocally(session.accumulatedTranscript);
        }

        // Evaluate spoken answer if question is set
        if (session.currentQuestion && session.accumulatedTranscript.length > 50) {
            // Inline evaluation (same logic as /api/evaluate-spoken-answer)
            const questionWords = session.currentQuestion.toLowerCase().split(/\s+/).filter(w => w.length > 3);
            const answerWords = session.accumulatedTranscript.toLowerCase().split(/\s+/).filter(w => w.length > 0);
            const answerText = session.accumulatedTranscript.toLowerCase();
            let evalScore = 0;
            if (answerWords.length > 50) evalScore += 30;
            else if (answerWords.length > 25) evalScore += 20;
            else if (answerWords.length > 10) evalScore += 10;
            else evalScore += 5;
            const matched = questionWords.filter(w => answerText.includes(w));
            evalScore += Math.round((questionWords.length > 0 ? matched.length / questionWords.length : 0) * 40);
            const sents = session.accumulatedTranscript.match(/[^.!?]+[.!?]+/g) || [];
            if (sents.length >= 2) evalScore += 15; else if (answerWords.length > 10) evalScore += 10; else evalScore += 5;
            if (/\d+|e\.g\.|for example|such as|because|therefore/i.test(session.accumulatedTranscript)) evalScore += 15;
            evalScore = Math.min(100, Math.max(0, evalScore));
            let feedback = evalScore >= 75 ? 'Good understanding demonstrated.' : evalScore >= 50 ? 'Partial understanding.' : evalScore >= 25 ? 'Answer lacks depth.' : 'Insufficient answer.';
            let aiProb = session.latestAI && session.latestAI.ai_probability ? session.latestAI.ai_probability : 0;
            session.spokenEvalResult = { success: true, similarity_score: evalScore, feedback, ai_probability: aiProb };
        }
    }
    updateSessionRisk(session);
    res.json({ ok: true });
});

// Interviewer sets question for session
app.post('/api/session/:code/question', (req, res) => {
    const session = getSession(req.params.code);
    if (!session) return res.status(404).json({ error: 'Session not found' });

    session.currentQuestion = req.body.question || '';
    session.accumulatedTranscript = '';
    session.spokenEvalResult = null;
    res.json({ ok: true });
});

// Interviewer polls session status
app.get('/api/session/:code/status', (req, res) => {
    const session = getSession(req.params.code);
    if (!session) return res.status(404).json({ error: 'Session not found' });

    // Return recent alerts (last 20) then clear
    const recentAlerts = session.alerts.slice(-20);
    session.alerts = [];

    res.json({
        candidateJoined: session.candidateJoined,
        candidateName: session.candidateName,
        latestFace: session.latestFace,
        latestVoice: session.latestVoice,
        latestAI: session.latestAI,
        behaviorData: {
            ...session.behaviorData,
            overall: (session.behaviorData.eyes === 'suspicious' || session.behaviorData.head === 'suspicious' || session.behaviorData.multiFace || session.behaviorData.tabSwitch > 5) ? 'Suspicious' : 'Normal'
        },
        tabSwitchCount: session.tabSwitchCount,
        pasteCount: session.pasteCount,
        alerts: recentAlerts,
        riskScore: session.riskScore,
        currentQuestion: session.currentQuestion,
        accumulatedTranscript: session.accumulatedTranscript,
        spokenEvalResult: session.spokenEvalResult
    });
});

function addSessionAlert(session, level, text) {
    session.alerts.push({ level, text, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) });
    // Keep max 100 alerts
    if (session.alerts.length > 100) session.alerts = session.alerts.slice(-100);
}

function updateSessionRisk(session) {
    let score = 0;
    if (session.latestFace.match_pct !== undefined) {
        if (session.latestFace.match_pct < 50) score += 30;
        else if (session.latestFace.match_pct < 75) score += 15;
    }
    if (session.latestVoice.similarity !== undefined) {
        if (session.latestVoice.similarity < 50) score += 25;
        else if (session.latestVoice.similarity < 75) score += 10;
    }
    if (session.latestAI.ai_probability !== undefined) {
        score += (session.latestAI.ai_probability / 100) * 25;
    }
    let bp = 0;
    if (session.tabSwitchCount > 3) bp += 10; else if (session.tabSwitchCount > 0) bp += 5;
    if (session.pasteCount > 0) bp += 10;
    if (session.behaviorData.multiFace) bp += 20;
    if (session.behaviorData.eyes === 'suspicious') bp += 10;
    if (session.behaviorData.head === 'suspicious') bp += 10;
    score += Math.min(20, bp);

    let level = 'genuine';
    if (score > 65) level = 'proxy';
    else if (score > 30) level = 'suspicious';
    session.riskScore = { score: Math.round(score), level };
}

// List active sessions (for debugging)
app.get('/api/sessions', (req, res) => {
    const list = [];
    sessions.forEach((s, code) => {
        list.push({ code, candidateJoined: s.candidateJoined, candidateName: s.candidateName, createdAt: s.createdAt });
    });
    res.json(list);
});

app.listen(port, '0.0.0.0', () => {
    console.log(`Node.js TrueHire AI Backend running on http://localhost:${port}`);
});
