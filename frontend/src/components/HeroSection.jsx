import React from 'react';
import { Circle, ArrowRight } from 'lucide-react';
import { SignInButton, useUser, useClerk } from '@clerk/clerk-react';
import { useNavigate } from 'react-router-dom';

const HeroSection = () => {
  const { user, isSignedIn } = useUser();
  const { signOut } = useClerk();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    await signOut();
    navigate('/');
  };

  return (
    <div className="flex flex-col space-y-8">
      <div className="pt-10 lg:pt-16">
        <div className="inline-flex items-center px-3 py-1 rounded-full bg-[#FFF1EC] border border-[#C8E19C]/20">
          <Circle className="h-4 w-4 mr-2 text-[#C5DD98] fill-[#C5DD98]" />
          <span className="text-sm font-medium text-gray-800">+20K cracked users</span>
        </div>
      </div>
      
      <div className="space-y-2">
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-gray-900 leading-tight">
        Axiom <br />
          <span className="text-[#51AC36]">Docs</span>
        </h1>
        
        <p className="text-lg text-gray-600 max-w-md mt-4">
        Transform your files into intelligent insights with AI-powered search and conversation - all in one secure cloud space
        </p>
      </div>
      
      <div className="flex flex-col sm:flex-row gap-4 pt-4">
        {!isSignedIn ? (
          <SignInButton mode="modal" afterSignInUrl="/welcome">
            <button className="px-6 py-3 bg-[#51AC36] text-white rounded-md hover:bg-[#51AC36] transition-colors duration-300 font-medium">
              Login / Signup
            </button>
          </SignInButton>
        ) : (
          <div className="flex items-center gap-4">
            <button 
              onClick={() => navigate('/welcome')}
              className="px-6 py-3 bg-[#51AC36] text-white rounded-md hover:bg-[#51AC36] transition-colors duration-300 font-medium"
            >
              Go to Dashboard
            </button>
            <button 
              onClick={handleSignOut}
              className="px-6 py-3 border border-gray-300 text-gray-600 rounded-md hover:bg-gray-50 transition-colors duration-300 font-medium"
            >
              Sign Out
            </button>
          </div>
        )}
        
        <button className="px-6 py-3 border border-gray-300 text-green-600 rounded-md hover:bg-gray-50 transition-colors duration-300 font-medium flex items-center justify-center">
          Docs
          <ArrowRight className="ml-2 h-4 w-4" />
        </button>
      </div>
      
      <div className="pt-16">
      </div>
    </div>
  );
};

export default HeroSection;