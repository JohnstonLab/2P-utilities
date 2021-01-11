
function b = i2cTest()
    hSI = evalin('base','hSI');             % get hSI from the base workspace
    assert(strcmpi(hSI.acqState,'idle')) ;   
    saveLocation = hSI.hScan2D.logFilePath;
    CurrentFileName = hSI.hScan2D.logFileStem;
    CurrentFrames = hSI.hStackManager.framesPerSlice;
    
    hSI.hScan2D.logFileStem = 'i2cTest'  ;    % set the base file name for the Tiff file
    hSI.hScan2D.logFileCounter = 1;
    hSI.hChannels.loggingEnable = true;
    hSI.hStackManager.framesPerSlice = 2;
    hSI.startGrab();
    pause(0.2);
    hSI.hStackManager.framesPerSlice = CurrentFrames
    hSI.hScan2D.logFileStem = CurrentFileName
    data = scanimage.util.opentif(strcat(saveLocation,'\i2cTest_00001_00001.tif'));
    delete(strcat(saveLocation,'\i2cTest_00001_00001.tif'));
    delete(strcat(saveLocation,'\i2cTest_Motion_00001.csv'));
    a =data.I2CData(1);
    b=a{1,1}{1,1};
    
    
end

