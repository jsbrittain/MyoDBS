#include "stdafx.h"
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <myo/myo.hpp>
#include <windows.h>

const int BUFFER_SAMPLES = 200;
const int PARAM_COUNT = (8 + 3 + 4);		// 8-EMG, 3-Acc, 4-Ori

using namespace std;
ofstream myfile;

class GraspDeterminator {
private:
	bool debug = true;
	int bufferLen = BUFFER_SAMPLES;
	int bufferPosEMG = 0;
	int bufferPosAcc = 0;
	int bufferPosOri = 0;
	int bufferPosProb = 0;
	int8_t **emgSamples;
	myo::Vector3<float> *accSamples;
	myo::Quaternion<float> *oriSamples;
	bool grasping = false;
	int acquiredEMGSamples = 0;
	int acquiredAccSamples = 0;
	// Beta parameters for logistic regression
	bool trained = false;
	int probSmoothing, stepbacks;
	float *beta;				// Excess parameters will simply not be employed
	float *bufferProb;
	int betacount;
	// Refractory period for stimulation switching
	float lastStimSwitchTime = 0;
public:

	// Constructor
	GraspDeterminator() {
		// Allocate memory arrays
		emgSamples = new int8_t*[BUFFER_SAMPLES];
		for (int k = 0; k < BUFFER_SAMPLES; k++) {
			emgSamples[k] = new int8_t[8];
			// Initialise
			for (int j = 0; j < 8; j++)
				emgSamples[k][j] = 0;
		}
		accSamples = new myo::Vector3<float>[BUFFER_SAMPLES];
		oriSamples = new myo::Quaternion<float>[BUFFER_SAMPLES];
		bufferProb = new float[BUFFER_SAMPLES];
		// Ensure probability buffer set to all zeros
		for (int k = 0; k < BUFFER_SAMPLES; k++)
			bufferProb[k] = 0;
	}

	// Destructor
	~GraspDeterminator() {
		for (int k = 0; k < BUFFER_SAMPLES; k++)
			delete[] emgSamples[k];
		delete[] emgSamples;
		delete[] accSamples;
		delete[] oriSamples;
		if (beta!=nullptr)
			delete[] beta;
	}

	// Dynamically allocate beta list
	void initBetaList(int paramcount) {
		// Delete old list first
		if (beta != nullptr)
			delete[] beta;
		// Allocate new memory
		beta = new float[paramcount];
		cout << "\nBeta memory allocated for " << paramcount << "elements\n";
	}
	
	// Log EMG data
	void addDataEMG(const int8_t* emg) {
		// Store for screen update
		bufferPosEMG = mod(bufferPosEMG + 1, BUFFER_SAMPLES);
		for (int i = 0; i < 8; i++) {
			emgSamples[bufferPosEMG][i] = abs( emg[i] );		// Store magnitude information only
		}
		acquiredEMGSamples += 1;
	}

	// Log accelerometer data
	void addDataAcc(myo::Vector3<float> accel) {
		// Store for screen update
		bufferPosAcc = mod(bufferPosAcc + 1, BUFFER_SAMPLES);		// Raw accelerometry
		accSamples[bufferPosAcc] = accel;
		acquiredAccSamples += 1;
	}

	// Log orientation data
	void addDataOri(myo::Quaternion<float> rotation) {
		// Store for screen update
		bufferPosOri = mod(bufferPosOri + 1, BUFFER_SAMPLES);		// Raw gyroscopic
		oriSamples[bufferPosOri] = rotation;
	}

	// Return grasping state
	bool isGrasping() {
		return trained && grasping;
	}

	bool loadTrainingParams(std::string filename) {
		std::string line;
		ifstream myfile(filename);
		if (myfile.is_open()) {
			// Read stepbacks first
			getline(myfile, line);
			stepbacks = (int) ::atof(line.c_str());		// Need to convert to float first for exponent format
			if (debug) cout << " Stepbacks = " << stepbacks << "\n";
			// Read probability smoothing scalar first
			getline(myfile, line);
			probSmoothing = (int) ::atof(line.c_str());
			if (debug) cout << " Probability smoothing = " << probSmoothing << "\n Beta = ";
			// Now read list of beta parameters
			betacount = PARAM_COUNT*stepbacks + 1;
			initBetaList(betacount);
			for (int k = 0; k < betacount; k++) {
				getline(myfile, line);
				beta[k] = ::atof(line.c_str());
				if (debug) cout << beta[k] << " ";
			}
			if (debug) cout << "\n";
		} else
			return trained = false;

		return trained = true;
	}

	void unloadTrainingParams() {
		trained = false;
		if (beta != nullptr)
			delete[] beta;
		return;
	}

	bool isTrained() {
		return trained;
	}

	int mod(int a, int b)
	{
		return (a%b + b) % b;
	}

	void updateGraspState() {
		// Only if trained
		if (!trained)
			return;
		// Update grasp state -- perform logistic regression on history
		if ((acquiredEMGSamples < stepbacks) || (acquiredAccSamples < stepbacks))
			return;
		
		// Intercept
		int ix = 0;
		float t = beta[ix++];
		// EMG channels
		int ch, k;
		for (ch = 0; ch < 8; ch++)
			for (k = 0; k < stepbacks; k++) {
				t += beta[ix++] * emgSamples[mod(bufferPosEMG - k,BUFFER_SAMPLES)][ch];
			}
		// Acc channels
		for (ch = 0; ch < 3; ch++)
			for (k = 0; k < stepbacks; k++)
				t += beta[ix++] * accSamples[mod(bufferPosAcc - k, BUFFER_SAMPLES)][ch];
		// Ori channels
		for (k = 0; k < stepbacks; k++)
			t += beta[ix++] * oriSamples[mod(bufferPosOri - k, BUFFER_SAMPLES)].w();
		for (k = 0; k < stepbacks; k++)
			t += beta[ix++] * oriSamples[mod(bufferPosOri - k, BUFFER_SAMPLES)].x();
		for (k = 0; k < stepbacks; k++)
			t += beta[ix++] * oriSamples[mod(bufferPosOri - k, BUFFER_SAMPLES)].y();
		for (k = 0; k < stepbacks; k++)
			t += beta[ix++] * oriSamples[mod(bufferPosOri - k, BUFFER_SAMPLES)].z();

		// Logit function
		bufferPosProb = mod(bufferPosProb + 1, BUFFER_SAMPLES);
		bufferProb[bufferPosProb] = 1 / (1 + exp(-t));
		grasping = getSmoothedProb() > 0.5;

		// Debug output
		if (false & (acquiredEMGSamples == 1000)) {
			ofstream debugfile("debug.txt");
			ix = 0;
			debugfile << "b(" << ix << ") = " << beta[ix] << "\n"; ix++;
			for (ch = 0; ch < 8; ch++)
				for (k = 0; k < stepbacks; k++) {
					debugfile << "b(" << ix << ")=" << beta[ix] << " " << bufferPosEMG << " " << bufferPosEMG - k << "," << BUFFER_SAMPLES << " [" << mod(bufferPosEMG - k, BUFFER_SAMPLES) << "][" << ch << "] " << (int)emgSamples[mod(bufferPosEMG - k, BUFFER_SAMPLES)][ch] << ": " << beta[ix] * emgSamples[mod(bufferPosEMG - k, BUFFER_SAMPLES)][ch] << "\n";
					ix++;
				}
			for (ch = 0; ch < 3; ch++)
				for (k = 0; k < stepbacks; k++) {
					debugfile << "b(" << ix << ")=" << beta[ix] << " " << bufferPosAcc << " " << bufferPosAcc - k << "," << BUFFER_SAMPLES << " [" << mod(bufferPosAcc - k, BUFFER_SAMPLES) << "][" << ch << "] " << accSamples[mod(bufferPosAcc - k, BUFFER_SAMPLES)][ch] << ": " << beta[ix] * accSamples[mod(bufferPosAcc - k, BUFFER_SAMPLES)][ch] << "\n";
					ix++;
				}
			for (k = 0; k < stepbacks; k++) {
				debugfile << "b(" << ix << ")=" << beta[ix++] << " " << oriSamples[mod(bufferPosOri - k, BUFFER_SAMPLES)].w() << "\n";
			}
			for (k = 0; k < stepbacks; k++)
				debugfile << "b(" << ix << ")=" << beta[ix++] << " " << oriSamples[mod(bufferPosOri - k, BUFFER_SAMPLES)].x() << "\n";
			for (k = 0; k < stepbacks; k++)
				debugfile << "b(" << ix << ")=" << beta[ix++] << " " << oriSamples[mod(bufferPosOri - k, BUFFER_SAMPLES)].y() << "\n";
			for (k = 0; k < stepbacks; k++)
				debugfile << "b(" << ix << ")=" << beta[ix++] << " " << oriSamples[mod(bufferPosOri - k, BUFFER_SAMPLES)].z() << "\n";
			debugfile << "\n t=" << t << " p=" << bufferProb[bufferPosProb];

			debugfile.close();
			myfile.close();

			std::string thingy;
			cin >> thingy;
			exit(1);
		}
	}

	float getSmoothedProb() {
		// Smooth probability vector and set grasp state
		float cumprob = 0;
		for (int k = 0; k < probSmoothing; k++)
			cumprob += bufferProb[mod(bufferPosProb - k, BUFFER_SAMPLES)];
		return cumprob /= probSmoothing;
	}

	float currentProb( ) {
		return bufferProb[bufferPosProb];
	}
};

class DataCollector : public myo::DeviceListener {
public:
	DataCollector() : emgSamples()
	{
	}

	GraspDeterminator grasp;

	std::array<int8_t, 8> emgSamples;
	int intAnnotation;
	std::string strAnnotationList[12];
	std::string strAnnotation;
	std::string strPose;
	int emgIndex = 0;

	void onUnpair(myo::Myo* myo, uint64_t timestamp)
	{
	}

	// onEmgData() is called whenever a paired Myo has provided new EMG data, and EMG streaming is enabled.
	void onEmgData(myo::Myo* myo, uint64_t timestamp, const int8_t* emg)
	{
		myfile << std::fixed << timestamp / 1e6 << "\tEMG";
		for (size_t i = 0; i < 8; i++) {
			std::ostringstream oss;
			oss << static_cast<int>(emg[i]);
			std::string emgString = oss.str();
			myfile << "\t" << (int)emg[i];
		}
		myfile << "\n";

		// Store for screen update
		for (int i = 0; i < 8; i++) {
			emgSamples[i] = emg[i];
		}

		// Log data to grasp determinator
		grasp.addDataEMG(emg);
		grasp.updateGraspState();
		if (grasp.isTrained())
			myfile << std::fixed << timestamp / 1e6 << "\tSTIM\t" << grasp.isTrained() << "\t" << grasp.currentProb() << "\t" << grasp.getSmoothedProb() << "\n";
	}

	void onAccelerometerData(myo::Myo* myo, uint64_t timestamp, const myo::Vector3<float> &accel)
	{
		myfile << std::fixed << timestamp/1e6 << "\tACC\t" << accel[0] << "\t" << accel[1] << "\t" << accel[2] << "\n";
		grasp.addDataAcc(accel);
	}

	void onGyroscopeData(myo::Myo* myo, uint64_t timestamp, const myo::Vector3<float> &gyro)
	{
		myfile << std::fixed << timestamp / 1e6 << "\tGYRO\t" << gyro[0] << "\t" << gyro[1] << "\t" << gyro[2] << "\n";
	}

	void onOrientationData(myo::Myo* myo, uint64_t timestamp, const myo::Quaternion<float> &rotation)
	{
		myfile << std::fixed << timestamp / 1e6 << "\tORI\t" << rotation.w() << "\t" << rotation.x() << "\t" << rotation.y() << "\t" << rotation.z() << "\n";
		grasp.addDataOri(rotation);
	}

	void onPose(myo::Myo* myo, uint64_t timestamp, myo::Pose &pose)
	{
		strPose = pose.toString();
		myfile << std::fixed << timestamp / 1e6 << "\tPOSE\t" << pose.toString() << "\n";
	}

	void onRSSI(myo::Myo* myo, uint64_t timestamp, int8_t &rssi)
	{
		myfile << std::fixed << timestamp / 1e6 << "\tRSSI\t" << rssi << "\n";
	}

	bool trainOnDataset( std::string filename )
	{
		// Implement training algorithm here //
		
		// Return true if trained, false if not
		return false;
	}
	
	void print()
	{
		// Clear the current line
		std::ostringstream oss, ossp;
		oss << static_cast<int>(emgSamples[0]);
		std::string emgString = oss.str();
		ossp << static_cast<int>(grasp.getSmoothedProb());
		std::string osspstring = ossp.str();
		if (grasp.isTrained())
			std::cout << "\r EMG-1 = [" << emgString << std::string(4-emgString.size(),' ') << "], Stim = " << grasp.isGrasping() << " (p = " << osspstring << std::string(8 - osspstring.size(), ' ') << "), State = " << strAnnotation << "       ";
		else
			std::cout << "\r EMG-1 = [" << emgString  << std::string(4-emgString.size(), ' ') << "], (Stim OFF), State = " << strAnnotation << "       ";
		std::cout << std::flush;
	}

	void readAnnotations()
	{
		std::string line;
		ifstream myfile("annot.txt");
		if (myfile.is_open())
		{
			for (int k = 0; k < 12; k++ )
				getline(myfile, strAnnotationList[k]);
			myfile.close();
		}
		else {
			cout << "Unable to open annotations definition file.";
			exit(-1);
		}
		return;
	}

	std::string getAnnotation(int annot)
	{
		return strAnnotationList[annot-1];
	}

	std::string setAnnotation(int annot)
	{
		if (annot == intAnnotation) {
			return strAnnotation;
		}
		strAnnotation = getAnnotation(annot);
		intAnnotation = annot;
		
		// Record to file
		myfile << std::fixed << 0 << "\tANNOT\t" << 'F' << annot << ' ' << strAnnotation << "\n";

		return strAnnotation;
	}

	std::string  GetFileName(const string & prompt) {
		const int BUFSIZE = 1024;
		char buffer[BUFSIZE] = { 0 };
		OPENFILENAME ofns = { 0 };
		ofns.lStructSize = sizeof(ofns);
		ofns.lpstrFile = buffer;
		ofns.nMaxFile = BUFSIZE;
		ofns.lpstrTitle = prompt.c_str();
		GetOpenFileName(&ofns);
		return buffer;
	}

	bool loadTrainingParams(std::string filename) {
		return grasp.loadTrainingParams(filename);
	}

	void unloadTrainingParams() {
		grasp.unloadTrainingParams();
		return;
	}

	void simulateInput(myo::Myo* myo, std::string filename) {
		std::string line;
		ifstream simfile;

		simfile.open(filename);
		if (simfile.is_open()) {
			// Output filename
			filename = filename + "_sim.txt";
			myfile.open(filename);
			// Read contents
			while (getline(simfile, line)) {
				char delimiter = '\t';
				size_t pos = 0;
				std::string token;

				// Time
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				line.erase(0, pos + 1);
				uint64_t timestamp = (uint64_t)(::atof(token.c_str())*1e6);

				// Identifier
				pos = line.find(delimiter);
				std::string ident = line.substr(0, pos);
				line.erase(0, pos + 1);

				// Data
				if (ident.compare("EMG") == 0) {
					int8_t emg[8];
					for (int k = 0; k < 8; k++) {
						pos = line.find(delimiter);
						token = line.substr(0, pos);
						line.erase(0, pos + 1);
						emg[k] = (int8_t) ::atoi(token.c_str());
					}
					onEmgData(myo, timestamp, emg);
				}
				if (ident.compare("ACC") == 0) {
					float acc[3];
					for (int k = 0; k < 3; k++) {
						pos = line.find(delimiter);
						token = line.substr(0, pos);
						line.erase(0, pos + 1);
						acc[k] = ::atof(token.c_str());
					}
					myo::Vector3<float> accel(acc[0], acc[1], acc[2]);
					onAccelerometerData(myo, timestamp, accel);
				}
				if (ident.compare("ORI") == 0) {
					float quart[4];
					for (int k = 0; k < 4; k++) {
						pos = line.find(delimiter);
						token = line.substr(0, pos);
						line.erase(0, pos + 1);
						quart[k] = ::atof(token.c_str());
					}
					myo::Quaternion<float> rotation(quart[1], quart[2], quart[3], quart[0]);		// { x, y, z, w }
					onOrientationData(myo, timestamp, rotation);
				}
				if (ident.compare("GYRO") == 0) {
					float gyr[3];
					for (int k = 0; k < 3; k++) {
						pos = line.find(delimiter);
						token = line.substr(0, pos);
						line.erase(0, pos + 1);
						gyr[k] = ::atof(token.c_str());
					}
					myo::Vector3<float> gyro(gyr[0], gyr[1], gyr[2]);
					onGyroscopeData(myo, timestamp, gyro);
				}
				if (ident.compare("ANNOT") == 0) {
					pos = line.find(' ');
					token = line.substr(1, pos);
					int fkey = ::atoi(token.c_str());
					setAnnotation(fkey);
				}

				// Update display
				print();
				// Break condition
				if (GetAsyncKeyState(VK_ESCAPE) & 0x8000)
					break;
			}
			myfile.close();
			simfile.close();
		}
		else
			cout << "Unable to open file!\n\n";
	}

};

int main(int argc, char** argv)
{
	// We catch any exceptions that might occur below -- see the catch statement for more details.
	try {
		// First, we create a Hub with our application identifier. Be sure not to use the com.example namespace when
		// publishing your application. The Hub provides access to one or more Myos.
		//myo::Hub hub("com.myodbs.myodbs");
		myo::Hub hub("com.example.emg-data-sample");
		std::cout << "Attempting to find a Myo..." << std::endl;
		
		// waitForMyo() takes a timeout value in milliseconds, in this case we will try to find a Myo for 10 seconds, and
		myo::Myo* myo = hub.waitForMyo(10000);
		if (!myo) {
			throw std::runtime_error("Unable to find a Myo!");
		}
		std::cout << "Connected to a Myo armband!" << std::endl << std::endl;
		
		// Next we enable EMG streaming on the found Myo.
		myo->setStreamEmg(myo::Myo::streamEmgEnabled);
		
		// Next we construct an instance of our DeviceListener, so that we can register it with the Hub.
		DataCollector collector;
		hub.addListener(&collector);

		// Read and report annotation keys
		collector.readAnnotations();
		cout << "Annotation keys F1-F12\n";
		for (int k = 1; k <= 12; k++) {
			cout << ' ' << collector.getAnnotation(k) << '\n';
		}

		enum States { state_menu, state_acquire, state_loadtrain, state_unloadtrain, state_train, state_exit, state_sim, state_vibrate };
		States state = state_menu;
		std::string filename, armside;
		int ch; int annot = 0;
		while (state != state_exit) {

			switch (state) {
			case state_menu:
				cout << "\nMenu\n 1) Acquire data\n 2) Load training file\n 3) Unload training data\n 4) Train using recorded dataset\n 5) Simulate recording\n 6) Vibrate Myo\n 7) Quit\n:";
				// Get option and flush buffer
				fflush(stdin);
				ch = getchar();
				fflush(stdin);
				switch (ch) {
				case '1':
					state = state_acquire;
					break;
				case '2':
					state = state_loadtrain;
					break;
				case '3':
					state = state_unloadtrain;
					break;
				case '4':
					state = state_train;
					break;
				case '5':
					state = state_sim;
					break;
				case '6':
					state = state_vibrate;
					break;
				case '7':
					state = state_exit;
					break;
				default:
					cout << "\n\n--- Unknown option ---\n";
				}
				break;

			case state_acquire:
				// Display annotation keys
				cout << "Annotation keys F1-F12\n";
				for (int k = 1; k <= 12; k++) {
					cout << ' ' << collector.getAnnotation(k) << '\n';
				}
				collector.setAnnotation(1);

				// Open file for data streaming
				cout << "Enter filename: ";
				cin >> filename;
				filename = "data/" + filename + ".txt";
				myfile.open(filename);
				// Patient details
				while (true) {
					if ((armside.compare("l") != 0) && (armside.compare("l") != 0)) {		// Only asks the first time!
						cout << "Which arm is the device on (l/r): ";
						cin >> armside;
					}
					if (armside.compare("l") == 0) {
						myfile << "0 PARAM ARMSIDE LEFT\n";
						break;
					}
					if (armside.compare("r") == 0) {
						myfile << "0 PARAM ARMSIDE RIGHT\n";
						break;
					};
				}

				// Main acquisition loop
				cout << "\nAquiring data...\n press numpad <1,2,3> to vibrate\n press ESC to finish.\n";
				while (true) {
					// In each iteration of our main loop, we run the Myo event loop for a set number of milliseconds.
					// In this case, we wish to update our display 50 times a second, so we run for 1000/20 milliseconds.
					hub.run(1000 / 20);
					collector.print();
					if (GetAsyncKeyState(VK_ESCAPE) & 0x8000)
						break;

					if (GetAsyncKeyState(VK_NUMPAD1) & 0x8000)
						myo->vibrate(myo::Myo::VibrationType::vibrationShort);
					if (GetAsyncKeyState(VK_NUMPAD2) & 0x8000)
						myo->vibrate(myo::Myo::VibrationType::vibrationMedium);
					if (GetAsyncKeyState(VK_NUMPAD3) & 0x8000)
						myo->vibrate(myo::Myo::VibrationType::vibrationLong);

					// Annotations
					for (int key = VK_F1; key <= VK_F12; key++) {
						if (GetAsyncKeyState(key) & 0x8000)
							collector.setAnnotation(key - VK_F1 + 1);
					}
				}
				// Close file
				myfile.close();
				// Tidy up menu
				cout << "\n";
				state = state_menu;
				break;

			case state_loadtrain:
				cout << "\n\nSpecify training parameters file...\n";
				filename = collector.GetFileName("Training parameters:");
				collector.loadTrainingParams(filename);
				state = state_menu;
				break;

			case state_unloadtrain:
				collector.unloadTrainingParams();
				break;

			case state_train:
				// Run Matlab to analyse dataset
				cout << "\n\nDo this in Matlab for now!\n\n";
				state = state_menu;
				break;

			case state_sim:
				// Simulate run using recorded data (tests online performance with known data)
				cout << "\n\n Simulate online environment\n\n";
				filename = collector.GetFileName("Simulation file:");
				collector.simulateInput(myo,filename);
				state = state_menu;
				break;

			case state_vibrate:
				// Test vibration functionality of the Myo
				myo->vibrate(myo::Myo::VibrationType::vibrationShort);
				Sleep(1000);
				myo->vibrate(myo::Myo::VibrationType::vibrationMedium);
				Sleep(1000);
				myo->vibrate(myo::Myo::VibrationType::vibrationLong);
				state = state_menu;
				break;

			default:
				cout << "\n\nCRITICAL ERROR --- UNRECOGNISED OPERATING MODE --- EXITING NOW ---";
				exit(0);
			}
		}
	}
	catch (const std::exception& e) {
		myfile.close();
		std::cerr << "Error: " << e.what() << std::endl;
		std::cerr << "Press enter to continue.";
		std::cin.ignore();
		return 1;
	}

	std::cout << "Closing gracefully...\ndone.";
	return 0;
}
