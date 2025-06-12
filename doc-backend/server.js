const { spawn } = require('child_process');
const express = require('express')
const multer = require('multer')
const cors = require('cors')
const path = require('path')
const { exec } = require('child_process')

const app = express()
const PORT = 3001

app.use(cors())
app.use(express.json())

// File upload config
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'uploads/'),
  filename: (req, file, cb) => cb(null, Date.now() + '-' + file.originalname)
})
const upload = multer({ storage })

// Upload route
app.post('/upload', upload.array('documents'), (req, res) => {
  console.log('Uploaded files:', req.files.map(f => f.filename))
  res.json({ message: 'Files uploaded successfully.' })
})

// Ask route
app.post('/ask', (req, res) => {
  const question = req.body.question;
  const folderPath = path.join(__dirname, 'uploads');

  const python = spawn('python', ['rag.py', folderPath, question]);

  let data = '';
  let error = '';

  python.stdout.on('data', (chunk) => {
    data += chunk.toString();
  });

  python.stderr.on('data', (err) => {
    error += err.toString();
  });

  python.on('close', (code) => {
    if (code !== 0) {
      console.error(`Python error (exit code ${code}):`, error);
      return res.status(500).json({ answer: "Error processing your question." });
    }
    let json;
    try {
      json = JSON.parse(data);
      res.json(json);
    } catch (err) {
      console.error('Failed to parse JSON:', err);
      res.json({ answer: data.trim() }); // fallback
    }

  });
});
app.listen(PORT, () => {
  console.log(`✅ Server is running on http://localhost:${PORT}`);
});

