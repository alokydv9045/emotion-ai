import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Box, 
  CssBaseline, 
  ThemeProvider, 
  createTheme, 
  AppBar, 
  Toolbar, 
  Typography, 
  Paper, 
  Grid, 
  Button, 
  LinearProgress,
  Divider,
  Chip,
  Alert
} from '@mui/material';
import { 
  Camera, 
  Zap, 
  Activity, 
  BarChart2, 
  Settings, 
  History, 
  Maximize2, 
  RefreshCcw, 
  Cpu, 
  Database,
  ShieldCheck,
  LayoutDashboard,
  WifiOff,
  Radio,
  Volume2,
  VolumeX,
  Mic,
  MicOff
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const API_URL = `${API_BASE_URL}/predict`;
const SPEAK_URL = `${API_BASE_URL}/speak`;
const POLLING_RATE = 200; 

const HINDI_PHRASES = {
  Happy: ["आपकी मुस्कान देखकर बहुत अच्छा लगा!", "वाह! आपकी खुशी संक्रामक है।", "मुस्कुराते रहिए, आप अच्छे लग रहे हैं।"],
  Sad: ["क्या हुआ? आप थोड़े उदास लग रहे हैं।", "चिंता न करें, सब ठीक हो जाएगा।", "मैं देख सकता हूँ कि आप थोड़े दुखी हैं।"],
  Angry: ["शांत हो जाइए, गुस्सा सेहत के लिए अच्छा नहीं है।", "गहरी साँस लें, सब ठीक है।", "गुस्सा छोड़िए और मुस्कुराइए।"],
  Surprise: ["ओह! यह तो काफी चौंकाने वाला है।", "हैरान लग रहे हैं आप?", "क्या बात है! आप तो चौंक गए।"],
  Fear: ["घबराइए मत, मैं यहाँ हूँ।", "डरने की कोई बात नहीं है।", "शांत रहें, सब नियंत्रण में है।"],
  Disgust: ["हम्म, लगता है आपको यह पसंद नहीं आया।", "कमाल है! आप इस पर विश्वास नहीं कर रहे?", "अजीब लग रहा है न?"],
  Neutral: ["आप काफी शांत लग रहे हैं।", "एकदम स्थिर और संतुलित।", "आपके चेहरे पर शांति दिख रही है।"]
};


// ── CUSTOM THEME: CYBER CYAN ──
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#22d3ee' },
    secondary: { main: '#0f172a' },
    background: { default: '#02040a', paper: '#090e1a' },
    text: { primary: '#f8fafc', secondary: '#64748b' },
  },
  typography: {
    fontFamily: '"Inter", "Outfit", sans-serif',
    h6: { fontWeight: 700, letterSpacing: '0.5px' },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          border: '1px solid rgba(34, 211, 238, 0.05)',
        }
      }
    }
  }
});

// ── SUB-COMPONENT: NEURAL BAR (CYBER VERSION) ──
const NeuralBar = ({ label, value, isActive }) => (
  <Box sx={{ 
    mb: 1.5, 
    p: isActive ? 0.8 : 0, 
    borderRadius: 1,
    bgcolor: isActive ? 'rgba(34, 211, 238, 0.05)' : 'transparent',
    transition: 'all 0.3s ease'
  }}>
    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5, px: 0.5 }}>
      <Typography 
        variant="caption" 
        sx={{ 
          color: isActive ? 'primary.main' : 'text.secondary', 
          fontWeight: isActive ? 800 : 600, 
          fontSize: '0.68rem',
          letterSpacing: isActive ? '1px' : 'normal',
          textShadow: isActive ? '0 0 8px rgba(34, 211, 238, 0.5)' : 'none',
          transition: 'all 0.3s ease'
        }}
      >
        {label.toUpperCase()}
      </Typography>
      <Typography 
        variant="caption" 
        sx={{ 
          color: isActive ? '#fff' : 'text.secondary', 
          fontSize: '0.68rem',
          fontWeight: isActive ? 700 : 400,
          transition: 'all 0.3s ease'
        }}
      >
        {value.toFixed(1)}%
      </Typography>
    </Box>
    <Box className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-900">
      <motion.div 
        className={`h-full ${isActive ? 'bg-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.8)]' : 'bg-slate-800'}`}
        initial={{ width: 0 }}
        animate={{ width: `${value}%` }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      />
    </Box>
  </Box>
);

const WaveformVisualizer = ({ isSpeaking }) => (
  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, height: 20 }}>
    {[1, 2, 3, 4, 5].map((i) => (
      <motion.div
        key={i}
        animate={isSpeaking ? { height: [4, 16, 8, 20, 4] } : { height: 4 }}
        transition={{ repeat: Infinity, duration: 0.5, delay: i * 0.1 }}
        style={{ width: 3, backgroundColor: '#22d3ee', borderRadius: 2 }}
      />
    ))}
  </Box>
);

function App() {
  const webcamRef = useRef(null);
  const [mode, setMode] = useState('live');
  const [emotion, setEmotion] = useState("Neutral");
  const [confidence, setConfidence] = useState(0);
  const [scores, setScores] = useState({});
  const [isScanning, setIsScanning] = useState(false);
  const [faces, setFaces] = useState([]);
  const [latency, setLatency] = useState(0);
  const [imgMeta, setImgMeta] = useState({ w: 1280, h: 720 });
  const [staticImg, setStaticImg] = useState(null);
  const [logs, setLogs] = useState([{ id: 1, time: new Date().toLocaleTimeString(), msg: "SYSTEM_BOOT: SUCCESS" }, { id: 2, time: new Date().toLocaleTimeString(), msg: "NEURAL_CORE: ONLINE" }]);
  const [history, setHistory] = useState([]); // Advanced session telemetry
  const [sessionStats, setSessionStats] = useState({ faces: 0, peak: "None", uptime: 0 });
  const [isLinkHealthy, setIsLinkHealthy] = useState(true);
  const [consecutiveErrors, setConsecutiveErrors] = useState(0);

  // ── VOICE CORE ──
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const lastSpokenRef = useRef({ emotion: "", time: 0 });
  const stabilityCounterRef = useRef({ emotion: "", count: 0 });
  const [voiceError, setVoiceError] = useState(null);
  const audioRef = useRef(null);

  // ── CORE NEURAL ENGINE (HARDENED) ──
  const sensorLoop = useCallback(async () => {
    if (!isScanning) return;
    
    let imageSrc = null;
    if (mode === 'live' && webcamRef.current) {
      imageSrc = webcamRef.current.getScreenshot();
    } else if (mode === 'photo' && staticImg) {
      imageSrc = staticImg;
    }

    if (!imageSrc) return;

    const start = performance.now();
    try {
      const response = await axios.post(API_URL, { 
        image: imageSrc,
        stream: mode === 'live'
      });
      const { emotion: emo, confidence: conf, scores: scs, faces: fcs, img_width, img_height } = response.data;
      
      const rtt = Math.round(performance.now() - start);
      setLatency(rtt);
      setEmotion(emo);
      setConfidence(conf);
      setScores(scs);
      setFaces(fcs);
      setImgMeta({ w: img_width, h: img_height });
      setIsLinkHealthy(true);
      setConsecutiveErrors(0);

      // 3. Update Session Context (Advanced History)
      setHistory(prev => [...prev.slice(-499), { emotion: emo, confidence: conf, timestamp: Date.now() }]);
      setLogs(prev => [{ id: Date.now(), time: new Date().toLocaleTimeString(), msg: `SCAN: ${emo.toUpperCase()} (${(conf * 100).toFixed(1)}%)` }, ...prev.slice(0, 499)]);
      
      setSessionStats(prev => ({
        faces: prev.faces + 1,
        peak: conf > 0.8 ? emo : prev.peak,
        uptime: prev.uptime + 1
      }));

      if (mode === 'photo') setIsScanning(false);

      // 4. Integrated Voice Trigger (Stability Guard)
      if (isVoiceEnabled && !isSpeaking) {
        const now = Date.now();
        const stableThreshold = 3; 
        
        if (stabilityCounterRef.current.emotion === emo) {
          stabilityCounterRef.current.count++;
        } else {
          stabilityCounterRef.current = { emotion: emo, count: 1 };
        }

        if (stabilityCounterRef.current.count >= stableThreshold) {
          const cooldown = 8000; 
          const isNewEmotion = lastSpokenRef.current.emotion !== emo;
          const timeSinceLast = now - lastSpokenRef.current.time;

          if (conf > 40 && (isNewEmotion || timeSinceLast > cooldown)) {
             handleSpeak(emo);
             stabilityCounterRef.current.count = 0; 
          }
        }
      }

    } catch (error) {
       setConsecutiveErrors(prev => prev + 1);
       if (consecutiveErrors > 3) setIsLinkHealthy(false);
       setLogs(prev => [{ id: Date.now(), time: new Date().toLocaleTimeString(), msg: "LINK ERROR: RETRYING..." }, ...prev.slice(0, 499)]);
       throw error; 
    }
  }, [isScanning, mode, staticImg, consecutiveErrors, isVoiceEnabled, isSpeaking]);

  const speakLocalFallback = (text) => {
    console.warn("VOICE_GUARD: Switching to Local Fallback (ElevenLabs Blocked)");
    
    // Ensure the voice engine is cancelled before starting new one (avoids queuing)
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    
    // Attempt to find a Hindi voice specifically
    const voices = window.speechSynthesis.getVoices();
    const hindiVoice = voices.find(v => v.lang.includes('hi')) || voices[0];
    
    utterance.voice = hindiVoice;
    utterance.lang = 'hi-IN';
    utterance.pitch = 1.4; // Slightly higher for Akshita persona
    utterance.rate = 1.0;
    
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    
    window.speechSynthesis.speak(utterance);
  };

  const handleSpeak = async (emo) => {
    try {
      setIsSpeaking(true);
      setVoiceError(null);
      
      const phrases = HINDI_PHRASES[emo] || HINDI_PHRASES["Neutral"];
      const randomPhrase = phrases[Math.floor(Math.random() * phrases.length)];
      setLogs(prev => [{ id: Date.now(), time: new Date().toLocaleTimeString(), msg: `VOICE: "${randomPhrase}"` }, ...prev.slice(0, 499)]);

      // 1. Play Akshita Greeting if it's the very first trigger
      if (lastSpokenRef.current.time === 0) {
          console.log("VOICE_GUARD: Playing Akshita Greeting...");
          const greeting = new Audio("/akshita.mp3");
          audioRef.current = greeting;
          greeting.play().catch(e => {
              console.warn("Greeting play failed, trying fallback...", e);
              speakLocalFallback(randomPhrase);
          });
          lastSpokenRef.current = { emotion: emo, time: Date.now() };
          greeting.onended = () => setIsSpeaking(false);
          return;
      }

      // 2. Standard ElevenLabs Request
      try {
        const response = await axios.post(SPEAK_URL, { text: randomPhrase, emotion: emo });
        const audioBase64 = response.data.audio;

        if (audioRef.current) {
          audioRef.current.pause();
          audioRef.current = null;
        }

        const audio = new Audio(`data:audio/mp3;base64,${audioBase64}`);
        audioRef.current = audio;
        audio.onended = () => {
          setIsSpeaking(false);
          audioRef.current = null;
        };
        await audio.play();
        lastSpokenRef.current = { emotion: emo, time: Date.now() };
      } catch (apiErr) {
        // 3. Fallback to Browser TTS if API fails/blocked
        speakLocalFallback(randomPhrase);
        lastSpokenRef.current = { emotion: emo, time: Date.now() };
        
        const errMsg = apiErr.response?.data?.detail || "Voice Engine Blocked";
        if (errMsg.includes("unusual_activity")) {
            setVoiceError("API_BLOCK: Using Local Fallback Mode.");
        }
      }
    } catch (err) {
      console.error("Critical Voice Error:", err);
      setIsSpeaking(false);
    }
  };

  // ── IMMORTAL WATCHDOG ──
  useEffect(() => {
    let tid;
    const poll = async () => {
      try {
        if (isScanning) await sensorLoop();
      } catch (e) {
        // Recover automatically
      } finally {
        tid = setTimeout(poll, POLLING_RATE);
      }
    };
    poll();
    return () => clearTimeout(tid);
  }, [isScanning, sensorLoop]);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (re) => { 
        setStaticImg(re.target.result); 
        setMode('photo');
        setIsScanning(true); 
        setLogs(prev => [{ id: Date.now(), time: new Date().toLocaleTimeString(), msg: `LOADED: ${file.name.toUpperCase()}` }, ...prev.slice(0, 499)]);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', bgcolor: 'background.default', overflow: 'hidden' }}>
        {voiceError && (
            <Box sx={{ bgcolor: 'error.main', color: 'white', px: 2, py: 0.5, fontSize: '0.7rem', textAlign: 'center', fontWeight: 'bold', zIndex: 9999 }}>
                SYSTEM_ALERT: {voiceError}
            </Box>
        )}
        
        {/* ── HEADER (CYBER) ── */}
        <AppBar position="fixed" elevation={0} sx={{ bgcolor: 'rgba(2, 4, 10, 0.8)', backdropFilter: 'blur(10px)', borderBottom: '1px solid rgba(34, 211, 238, 0.1)' }}>
          <Toolbar variant="dense" sx={{ justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
              <ShieldCheck size={18} className="text-cyan-400" />
              <Typography variant="h6" sx={{ fontSize: '0.9rem', color: 'primary.main' }}>CORTEX-V<span className="text-slate-500 font-normal"> / STUDIO</span></Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Chip 
                  label={!isLinkHealthy ? "LINK LOST" : (isScanning ? "LINK ACTIVE" : "READY")} 
                  size="small" 
                  color={!isLinkHealthy ? "error" : (isScanning ? "primary" : "default")} 
                  className={!isLinkHealthy ? "animate-pulse" : ""}
                  sx={{ fontSize: '0.6rem', height: 20 }} 
                />
                <Typography variant="caption" sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>Engine: v5.0_APEX (96x96)</Typography>
            </Box>
          </Toolbar>
        </AppBar>

        <Box sx={{ display: 'flex', flex: 1, mt: '48px', overflow: 'hidden' }}>
          
          {/* ── LEFT SIDEBAR (FIXED) ── */}
          <Box sx={{ width: 280, flexShrink: 0, p: 2, borderRight: '1px solid rgba(34, 211, 238, 0.1)', bgcolor: '#02040a' }}>
            <Box sx={{ mb: 4 }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', letterSpacing: '2px', display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <LayoutDashboard size={12} /> TACTICAL HUB
              </Typography>
              <Box sx={{ display: 'flex', bgcolor: 'background.paper', p: 0.5, borderRadius: 2, mb: 2, border: '1px solid rgba(34, 211, 238, 0.05)' }}>
                <Button fullWidth size="small" variant={mode === 'live' ? 'contained' : 'text'} onClick={() => setMode('live')} sx={{ fontSize: '0.7rem' }}>LIVE</Button>
                <Button fullWidth size="small" variant={mode === 'photo' ? 'contained' : 'text'} onClick={() => setMode('photo')} sx={{ fontSize: '0.7rem' }}>PHOTO</Button>
              </Box>
              <Button 
                fullWidth 
                variant={isScanning ? "outlined" : "contained"} 
                color={isScanning ? "error" : (mode === 'live' ? "primary" : "secondary")}
                onClick={() => setIsScanning(!isScanning)}
                startIcon={isScanning ? <RefreshCcw size={14} className={mode === 'live' ? "animate-spin" : ""} /> : <Zap size={14} />}
                sx={{ mb: 1, boxShadow: isScanning ? 'none' : '0 0 20px rgba(34, 211, 238, 0.2)' }}
              >
                {mode === 'live' ? (isScanning ? "DISCONNECT" : "ENGAGE STREAM") : (isScanning ? "STOP UPDATE" : "RUN ANALYSIS")}
              </Button>
              {mode === 'photo' && (
                <Button fullWidth variant="outlined" component="label" sx={{ mt: 1, borderColor: 'rgba(34, 211, 238, 0.2)' }}>
                  SCAN FILE
                  <input type="file" hidden onChange={handleFileUpload} accept="image/*" />
                </Button>
              )}
              
              <Box sx={{ mt: 2, p: 2, bgcolor: isVoiceEnabled ? 'rgba(34, 211, 238, 0.05)' : 'transparent', borderRadius: 2, border: '1px solid rgba(34, 211, 238, 0.1)' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: isVoiceEnabled ? 'primary.main' : 'text.secondary', fontWeight: 600 }}>NEURAL VOICE</Typography>
                  <Button 
                    size="small" 
                    variant={isVoiceEnabled ? "contained" : "outlined"} 
                    onClick={() => {
                        if (isVoiceEnabled && audioRef.current) audioRef.current.pause();
                        setIsVoiceEnabled(!isVoiceEnabled);
                    }}
                    sx={{ minWidth: 0, p: 0.5, borderRadius: '50%' }}
                  >
                    {isVoiceEnabled ? <Volume2 size={14} /> : <VolumeX size={14} />}
                  </Button>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <WaveformVisualizer isSpeaking={isSpeaking} />
                  <Typography variant="caption" sx={{ fontSize: '0.6rem', color: isSpeaking ? 'primary.main' : 'slate.600' }}>
                    {isSpeaking ? "SPEAKING..." : (isVoiceEnabled ? "LISTENING..." : "VOICE_OFF")}
                  </Typography>
                </Box>
              </Box>
            </Box>

            {!isLinkHealthy && (
              <Alert severity="error" icon={<WifiOff size={16} />} sx={{ mb: 3, fontSize: '0.7rem', py: 0 }}>
                CRITICAL: Backend Unreachable. Checking Launcher...
              </Alert>
            )}

            <Divider sx={{ my: 3, opacity: 0.1 }} />

            <Box>
              <Typography variant="caption" sx={{ color: 'text.secondary', letterSpacing: '2px', display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <Activity size={12} /> ENGINE STATUS
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <div className="flex justify-between items-center">
                   <span className="text-[0.7rem] text-slate-500">Neural Latency</span>
                    <span className="text-[0.7rem] font-mono text-cyan-400">{latency}ms [67.1% PRECISION]</span>
                </div>
                <div className="flex justify-between items-center">
                   <span className="text-[0.7rem] text-slate-500">Core Health</span>
                   <span className="text-[0.7rem] font-mono text-green-500">STABLE</span>
                </div>
                 <div className="flex justify-between items-center">
                    <span className="text-[0.7rem] text-slate-500">Neural Detail</span>
                    <span className="text-[0.7rem] font-mono text-cyan-400">96x96 RGB</span>
                 </div>
              </Box>
            </Box>
          </Box>

          {/* ── CENTER VIEWPORT (FLUID) ── */}
          <Box component="main" sx={{ flex: 1, p: 2, display: 'flex', flexDirection: 'column', gap: 2, bgcolor: '#02040a', minWidth: 0 }}>
            <Paper elevation={0} sx={{ flex: 1, position: 'relative', overflow: 'hidden', bgcolor: '#000', borderRadius: 4, border: '1px solid rgba(34, 211, 238, 0.1)' }}>
              {mode === 'live' ? (
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={{ width: 1280, height: 720, facingMode: "user" }}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
              ) : (
                <Box sx={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {staticImg ? (
                    <img src={staticImg} style={{ width: '100%', height: '100%', objectFit: 'contain' }} alt="Static analysis" />
                  ) : (
                    <Box sx={{ textAlign: 'center', color: 'text.secondary' }}>
                      <Radio size={48} className="mx-auto mb-4 opacity-10 animate-pulse text-cyan-400" />
                      <Typography variant="body2" sx={{ letterSpacing: 2 }}>WAITING FOR NEURAL SOURCE</Typography>
                    </Box>
                  )}
                </Box>
              )}

              {/* HUD OVERLAY (CYBER) */}
              <Box className="absolute inset-0 pointer-events-none z-10 overflow-hidden">
                <div className="hud-vignette"></div>
                <div className="hud-scanline opacity-30"></div>
                {isScanning && (
                  <motion.div 
                    className="absolute w-full h-[1px] bg-cyan-400/60 shadow-[0_0_20px_rgba(34,211,238,0.7)]"
                    animate={{ top: ["0%", "100%", "0%"] }}
                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                  />
                )}
                
                {/* HUD: Identity Reticles */}
                <AnimatePresence>
                  {faces.map((face, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute border border-cyan-400"
                      style={{
                        left: `${(face[0] / imgMeta.w) * 100}%`,
                        top: `${(face[1] / imgMeta.h) * 100}%`,
                        width: `${(face[2] / imgMeta.w) * 100}%`,
                        height: `${(face[3] / imgMeta.h) * 100}%`
                      }}
                    >
                      <div className="absolute -top-1 -left-1 w-1.5 h-1.5 border-l-2 border-t-2 border-cyan-400"></div>
                      <div className="absolute -top-1 -right-1 w-1.5 h-1.5 border-r-2 border-t-2 border-cyan-400"></div>
                      <div className="absolute -bottom-1 -left-1 w-1.5 h-1.5 border-l-2 border-b-2 border-cyan-400"></div>
                      <div className="absolute -bottom-1 -right-1 w-1.5 h-1.5 border-r-2 border-b-2 border-cyan-400"></div>
                      <div className="absolute -top-6 left-0 bg-cyan-900/80 px-2 py-0.5 border-l-2 border-cyan-400 text-[9px] text-cyan-200 font-mono whitespace-nowrap">
                         ID_{idx.toString().padStart(4, '0')} // {emotion}
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </Box>
            </Paper>

            <Paper elevation={0} sx={{ height: 160, p: 2, bgcolor: 'background.paper', display: 'flex', gap: 3, borderTop: '1px solid rgba(34, 211, 238, 0.1)' }}>
               <Box sx={{ width: 140, flexShrink: 0 }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Database size={10} /> SESSION_METRICS
                  </Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', color: 'slate.500', lineHeight: 1.5, display: 'block' }}>
                    SCANS: {sessionStats.faces}<br/>
                    PEAK: {sessionStats.peak}<br/>
                    UPTIME: {Math.floor(sessionStats.uptime/60)}m {sessionStats.uptime%60}s
                  </Typography>
               </Box>
               <Box sx={{ flex: 1, overflowY: 'auto', borderLeft: '1px solid rgba(34, 211, 238, 0.05)', pl: 2, className: 'custom-scrollbar' }}>
                  <Typography variant="caption" sx={{ color: 'cyan.900', mb: 1, display: 'block', fontSize: '0.6rem' }}>NEURAL_FEED_STREAMING (DEPTH: 500)</Typography>
                  {logs.map((log) => (
                    <Typography key={log.id} variant="caption" display="block" sx={{ fontFamily: 'monospace', color: 'text.secondary', opacity: 0.8, fontSize: '0.7rem' }}>
                      <span className="text-cyan-800">[{log.time}]</span> {log.msg}
                    </Typography>
                  ))}
               </Box>
            </Paper>
          </Box>

          {/* ── RIGHT PANEL (FIXED) ── */}
          <Box sx={{ width: 320, flexShrink: 0, p: 2, borderLeft: '1px solid rgba(34, 211, 238, 0.1)', bgcolor: '#02040a' }}>
            <Box sx={{ mb: 4 }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', letterSpacing: '2px', display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                <BarChart2 size={12} /> TELEMETRY OUTPUT
              </Typography>
              
              <Paper elevation={0} sx={{ p: 2, mb: 3, borderLeft: '4px solid', borderColor: 'primary.main', bgcolor: 'rgba(34, 211, 238, 0.05)', boxShadow: '0 0 20px rgba(34, 211, 238, 0.05)' }}>
                 <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.6rem' }}>NEURAL MODE SIGNATURE</Typography>
                 <Typography variant="h4" sx={{ mb: 0.5, fontFamily: 'Space Grotesk', color: 'primary.main' }}>{emotion.toUpperCase()}</Typography>
                 <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 600 }}>{confidence.toFixed(1)}% Match</Typography>
              </Paper>

              <Box sx={{ mt: 2 }}>
                {Object.entries(scores || {}).map(([emo, val]) => (
                  <NeuralBar key={emo} label={emo} value={val} isActive={emo === emotion} />
                ))}
              </Box>
            </Box>

            <Box sx={{ mt: 'auto', textAlign: 'center' }}>
               <Paper elevation={0} sx={{ p: 1, bgcolor: 'rgba(34, 211, 238, 0.02)', border: '1px solid rgba(34, 211, 238, 0.05)' }}>
                  <Typography variant="caption" sx={{ color: 'slate.700', fontSize: '0.55rem', letterSpacing: 2 }}>
                    CORTEX-V // PEAK STABILITY // ACTIVE
                  </Typography>
               </Paper>
            </Box>
          </Box>

        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
