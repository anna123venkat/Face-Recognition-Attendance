import unittest
from app.main import app

class TestApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup test client for Flask app
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_home_page(self):
        """Test if the home page loads correctly"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Face Recognition Based Attendance System', response.data)

    def test_start_attendance(self):
        """Test if the start attendance page works"""
        response = self.client.get('/start')
        # Assuming the page redirects if no model is trained
        self.assertIn(b'There is no trained model', response.data)

    def test_add_user_page(self):
        """Test if the add user page is accessible"""
        response = self.client.get('/add')
        self.assertEqual(response.status_code, 405)  # POST-only route

    def test_add_user_functionality(self):
        """Test if adding a new user works"""
        response = self.client.post('/add', data={
            'newusername': 'TestUser',
            'newuserid': 12345
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'New user added successfully!', response.data)

if __name__ == '__main__':
    unittest.main()
