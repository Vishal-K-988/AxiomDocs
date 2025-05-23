import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ClerkProvider, SignedIn, SignedOut, RedirectToSignIn } from '@clerk/clerk-react';
import LandingPage from './components/LandingPage';
import WelcomePage from './components/WelcomePage';

if (!import.meta.env.VITE_CLERK_PUBLISHABLE_KEY) {
  throw new Error("Missing Clerk Publishable Key");
}

const clerkPubKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

function App() {
  return (
    // herer I'm using the clerk provider and adding some syling / ui  with respect to our website 
    <ClerkProvider 
      publishableKey={clerkPubKey}
      appearance={{
        baseTheme: undefined,
        unsafe_disableDevelopmentModeWarnings: true,
        variables: {
          colorPrimary: '#51AC36',
          colorText: '#1F2937',
          colorTextSecondary: '#4B5563',
          colorBackground: '#FFFFFF',
          colorInputBackground: '#F9FAFB',
          colorInputText: '#1F2937',
          borderRadius: '0.375rem',
        },
        elements: {
          formButtonPrimary: {
            backgroundColor: '#51AC36',
            '&:hover': {
              backgroundColor: '#429A2B',
            },
          },
          card: {
            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
          },
          headerTitle: {
            color: '#1F2937',
          },
          headerSubtitle: {
            color: '#4B5563',
          },
        },
      }}
    >
      <Router>
        <div className="App">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route
              path="/welcome"
              element={
                <>
                  <SignedIn>
                    <WelcomePage />
                  </SignedIn>
                  <SignedOut>
                    <RedirectToSignIn redirectUrl="/welcome" />
                  </SignedOut>
                </>
              }
            />
          </Routes>
        </div>
      </Router>
    </ClerkProvider>
  );
}

export default App;