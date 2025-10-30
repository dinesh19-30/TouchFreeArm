% ---
% Part 1: Build the 2-Joint Robot (Refined Industrial Design)
% ---
clear; clc; close all;

% --- NEW: SETUP PYTHON INTEGRATION (FIXED) ---
disp('Setting up Python environment...');

try
    % Set the Python version
    py_path = "C:\Users\shshi\AppData\Local\Programs\Python\Python311\python.exe";
    pyenv("Version", py_path);
    
    fprintf('MATLAB is using Python: %s\n', pyenv().Version);
    
    % Test for required modules
    py.importlib.import_module('cv2');
    py.importlib.import_module('mediapipe');
    disp('cv2 and mediapipe modules found.');
    
    % --- THIS IS THE FIX ---
    % Changed py.sys.path to py.sys.path() in two places
    if ~any(strcmp(py.sys.path(),'.'))
        py.sys.path().insert(int32(0), '.');
    end
    % --- END FIX ---
    
    % Call the Python initialization function
    success = py.hand_tracker.initialize_tracker();
    if ~success
        error('Could not initialize Python hand tracker. Check webcam and dependencies.');
    end
    
catch e
    fprintf(2, 'Error: %s\n', e.message);
    error('MATLAB failed to initialize Python. Make sure Python, OpenCV, and MediaPipe are installed and configured.');
end
disp('Python tracker initialized. Starting robot simulation...');
% --- END PYTHON SETUP ---


% (The robot-building code is unchanged)
robot = rigidBodyTree('DataFormat', 'struct');

baseLink = rigidBody('base_link');
baseJoint = rigidBodyJoint('base_joint', 'fixed');
basePose = trvec2tform([0 0 0.1]);
addVisual(baseLink, 'Cylinder', [0.15, 0.2], basePose);
addBody(robot, baseLink, 'base');

L1_length = 0.5; 
link1 = rigidBody('link1');
joint1 = rigidBodyJoint('joint1', 'revolute');
joint1.JointAxis = [0 0 1];
setFixedTransform(joint1, trvec2tform([0 0 0.2]));
link1.Joint = joint1;
joint1_visual_pose = trvec2tform([0 0 0]);
addVisual(link1, 'Cylinder', [0.1, 0.1], joint1_visual_pose);
link1_visual_pose = trvec2tform([L1_length/2 0 0]);
addVisual(link1, 'Box', [L1_length, 0.08, 0.08], link1_visual_pose);
addBody(robot, link1, 'base_link');

L2_length = 0.5;
link2 = rigidBody('link2');
joint2 = rigidBodyJoint('joint2', 'revolute');
joint2.JointAxis = [0 0 1];
setFixedTransform(joint2, trvec2tform([L1_length, 0, 0])); 
link2.Joint = joint2;
joint2_visual_pose = trvec2tform([0 0 0]);
addVisual(link2, 'Cylinder', [0.08, 0.08], joint2_visual_pose);
link2_visual_pose = trvec2tform([L2_length/2 0 0]);
addVisual(link2, 'Box', [L2_length, 0.06, 0.06], link2_visual_pose);
addBody(robot, link2, 'link1');

flange = rigidBody('flange');
flange_joint = rigidBodyJoint('flange_joint', 'fixed');
setFixedTransform(flange_joint, trvec2tform([L2_length, 0, 0]));
flange.Joint = flange_joint;
flange_visual_pose = trvec2tform([0 0 0]);
addVisual(flange, 'Cylinder', [0.05, 0.02], flange_visual_pose);
addBody(robot, flange, 'link2');

disp('Refined 2-Joint industrial robot created.');

% ---
% Part 2: Setup Visualization
% ---
q_current = homeConfiguration(robot); 

figure;
title('Live Hand-Controlled Robot Simulation');
show(robot, q_current);
hold on;
axis([-1.5 1.5 -1.5 1.5 -0.5 2]); 
view(30, 25); 
grid on;

% ---
% Part 3: Run the LIVE Animation Loop
% ---
disp('Starting continuous demo... Press Ctrl+C in Command Window to stop.');

% Use an 'onCleanup' object to automatically call the Python shutdown
cleanupObj = onCleanup(@() py.hand_tracker.shutdown_tracker());

while true 
    
    % --- 1. GET ANGLES FROM PYTHON ---
    angles_py = py.hand_tracker.get_hand_angles();
    
    % Convert from Python float to MATLAB double
    angle_j1 = double(angles_py{1}); % L-SERVO 1
    angle_j2 = double(angles_py{2}); % R-SERVO 1
    
    % --- 2. UPDATE ROBOT CONFIGURATION ---
    % Map 0-180 degree angle to radians for MATLAB
    q_current(1).JointPosition = deg2rad(angle_j1); % Joint 1
    q_current(2).JointPosition = deg2rad(angle_j2); % Joint 2
    
    % --- 3. SHOW ROBOT ---
    show(robot, q_current, 'PreservePlot', false);
    title(['Live Control | J1 (Left): ', num2str(angle_j1, '%.0f'), ...
           ' deg | J2 (Right): ', num2str(angle_j2, '%.0f'), ' deg']);
    drawnow;
    
end