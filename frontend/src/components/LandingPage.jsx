import React from 'react';
import Navbar from './Navbar';
import HeroSection from './HeroSection';
import BackgroundGradient from './BackgroundGradient';

const LandingPage=  () => {
  return (
    <div className="relative w-full min-h-screen overflow-hidden bg-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <Navbar />
        <div className="flex flex-col lg:flex-row">
          <div className="lg:w-1/2 pt-8 lg:pt-16 pb-16">
            <HeroSection />
          </div>
          <div className="lg:w-1/2"></div>
        </div>
      </div>
      <BackgroundGradient />
      <div className='relative z-10'>

     
      <footer className="w-full py-10  bg-white border-t border-gray-100 text-center mt-10 z-10 relative">
        <span className="text-gray-500 pt-8 font-manrope text-base">Vishal Kumar Geed</span>
        <br />
      <br />
    
      </footer>
  </div>
    </div>
  );
};

export default LandingPage;