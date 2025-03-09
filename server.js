const express = require('express');
const bodyParser = require('body-parser');
const { MongoClient } = require('mongodb');
const path = require('path');

const app = express();
const PORT = 3000;

// MongoDB Connection
const MONGO_URI = 'mongodb://127.0.0.1:27017';
const DATABASE_NAME = 'diaryAppDB';

let db;
MongoClient.connect(MONGO_URI, { useUnifiedTopology: true })
    .then((client) => {
        db = client.db(DATABASE_NAME);
        console.log(`Connected to MongoDB: ${DATABASE_NAME}`);
    })
    .catch((err) => console.error('Failed to connect to MongoDB:', err));

// Middleware
app.use(bodyParser.json());
app.use(express.static(__dirname)); // Serve static files like HTML

// Serve `home.html` after login
app.get('/home.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'home.html'));
});

// Serve `diaryentry.html`
app.get('/diaryentry.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'diaryentry.html'));
});

// Signup Endpoint
app.post('/signup', async (req, res) => {
    const { email, password, confirmPassword, profession, areaOfInterest } = req.body;

    if (!email || !password || !confirmPassword || password !== confirmPassword) {
        return res.status(400).json({ message: 'Invalid input or passwords do not match' });
    }

    try {
        const existingUser = await db.collection('users').findOne({ email });
        if (existingUser) {
            return res.status(400).json({ message: 'User already exists' });
        }

        await db.collection('users').insertOne({ email, password, profession, areaOfInterest });
        res.status(200).json({ message: 'Signup successful' });
    } catch (error) {
        console.error('Error during signup:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Login Endpoint
app.post('/login', async (req, res) => {
    const { email, password } = req.body;

    if (!email || !password) {
        return res.status(400).json({ success: false, message: 'Email and password are required' });
    }

    try {
        const user = await db.collection('users').findOne({ email });

        if (!user || user.password !== password) {
            return res.status(401).json({ success: false, message: 'Invalid email or password' });
        }

        res.status(200).json({ success: true, message: 'Login successful' });
    } catch (error) {
        console.error('Error during login:', error);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Check if an entry exists for the given date
app.get('/checkEntry', async (req, res) => {
  const date = req.query.date;
  const searchDate = new Date(date);
  searchDate.setHours(0, 0, 0, 0);
  const nextDay = new Date(searchDate.getTime() + 24 * 60 * 60 * 1000);

  try {
    const existingEntry = await db.collection('entries').findOne({
      date: { $gte: searchDate, $lt: nextDay },
    });

    if (existingEntry) {
      res.json({ exists: true, entry: existingEntry.entry });
    } else {
      res.json({ exists: false });
    }
  } catch (error) {
    console.error('Error checking entry:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

app.post('/saveEntry', async (req, res) => {
  const { date, entry } = req.body;

  if (!date || !entry) {
    return res.status(400).json({ message: 'Date and entry content are required' });
  }

  try {
    // Save the new entry with a unique timestamp or ID
    const result = await db.collection('entries').insertOne({
      date: new Date(date),
      entry,
      timestamp: new Date(), // Optional: Add a timestamp for sorting later
    });

    res.status(200).json({ message: 'Entry saved successfully', entryId: result.insertedId });
  } catch (error) {
    console.error('Error saving entry:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});


app.post('/updateEntry', async (req, res) => {
  console.log('Update entry endpoint called');
  const { date, entry } = req.body;

  if (!date || !entry) {
    return res.status(400).json({ message: 'Date and entry content are required' });
  }

  try {
    const searchDate = new Date(date);
    searchDate.setHours(0, 0, 0, 0); // Normalize to midnight
    const nextDay = new Date(searchDate.getTime() + 24 * 60 * 60 * 1000);

    // Find the existing entry
    const existingEntry = await db.collection('entries').findOne({
      date: { $gte: searchDate, $lt: nextDay },
    });

    if (!existingEntry) {
      return res.status(404).json({ message: 'No entry found for the given date' });
    }

    console.log('Found existing entry:', existingEntry);

    // Avoid duplicating content
    const existingText = existingEntry.entry.trim();
    const newText = entry.trim();
    console.log('Existing entry:',existingText)
    console.log('New Text:',newText)
    // Check if the new text is already part of the existing text
    const updatedEntry = existingText.includes(newText)
      ? existingText
      : `${existingText} ${newText}`.trim();

    console.log('Updated Entry:', newText);

    // Delete the old entry
    const deleteResult = await db.collection('entries').deleteOne({
      _id: existingEntry._id, // Use unique _id to avoid errors
    });
    console.log(`Deleted entry with _id: ${existingEntry._id}`);

    if (deleteResult.deletedCount === 0) {
      return res.status(500).json({ message: 'Failed to delete the existing entry.' });
    }

    // Save the updated entry
    const insertResult = await db.collection('entries').insertOne({
      date: searchDate,
      entry: newText,
      timestamp: new Date(),
    });

    console.log('New entry saved with _id:', insertResult.insertedId);

    res.status(200).json({ message: 'Entry updated successfully', entry: updatedEntry });
  } catch (error) {
    console.error('Error updating entry:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Fetch Diary Entries Endpoint
app.get('/fetch-entry', async (req, res) => {
  const { date } = req.query;

  if (!date) {
      return res.status(400).json({ message: 'Date is required' });
  }

  try {
      const searchDate = new Date(date);
      if (isNaN(searchDate)) {
          return res.status(400).json({ message: 'Invalid date format' });
      }

      searchDate.setHours(0, 0, 0, 0);

      const nextDay = new Date(searchDate.getTime() + 24 * 60 * 60 * 1000);

      // Fetch all entries for the date
      const entries = await db.collection('entries').find({
          date: { $gte: searchDate, $lt: nextDay },
      }).project({entry :1,_id:0}).toArray();
      console.log('Fetched entries:', entries); // Debugging
      if (entries.length > 0) {
          res.json({ entries });
      } else {
          res.json({ entries: [] });
      }
  } catch (error) {
      console.error('Error fetching entries:', error);
      res.status(500).json({ message: 'Internal server error' });
  }
});




// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
