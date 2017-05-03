package application;

import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import org.apache.commons.math3.util.FastMath;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;


public class FaceDetectionController
{
	// FXML buttons
	@FXML
	private Button cameraButton;
	// the FXML area for showing the current frame
	@FXML
	private ImageView originalFrame;
	
	
	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that performs the video capture
	private VideoCapture capture;
	// a flag to change the button behavior
	private boolean cameraActive;
	
	// face cascade classifier
	private CascadeClassifier faceCascade;
	private int sample_count;
	
	
	//DataBuffer to store values
	ArrayList<Double> Val_Buffer;
	ArrayList<Double> Time_Buffer; 
	
	Timestamp starttime;
	Timestamp endtime;
	int prev_x,prev_y,start;
	/**
	 * Init the controller, at start time
	 */
	protected void init()
	{

		this.start = 1;
		this.Val_Buffer = new ArrayList<Double>();
		this.Time_Buffer = new ArrayList<Double>();
		
		this.capture = new VideoCapture();
		this.faceCascade = new CascadeClassifier();
		this.sample_count = 128;
		originalFrame.setFitWidth(600);
		// preserve image ratio
		originalFrame.setPreserveRatio(true);
		checkboxSelection();
	}
	
	/**
	 * The action triggered by pushing the button on the GUI
	 */
	@FXML
	protected void startCamera()
	{	
		if (!this.cameraActive)
		{
	
			
			// start the video capture
			this.capture.open("2.avi");
			//this.capture.open(0);
			
			// is the video stream available?
			if (this.capture.isOpened())
			{
				this.cameraActive = true;
				
				// grab a frame every 100 ms (100 frames/sec)
				Runnable frameGrabber = new Runnable() {
					
					@Override
					public void run()
					{
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						updateImageView(originalFrame, imageToShow);
					}
				};
				
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 100, TimeUnit.MILLISECONDS);
				
				// update the button content
				this.cameraButton.setText("Stop Analysis");
			}
			else
			{
				// log the error
				System.err.println("Failed to open the camera connection...");
			}
		}
		else
		{
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.cameraButton.setText("Start Analysis");
			
			// stop the timer
			this.stopAcquisition();
		}
	}
	
	/**
	 * Get a frame from the opened video stream (if any)
	 * 
	 * @return the {@link Image} to show
	 */
	private Mat grabFrame()
	{
		Mat frame = new Mat();
		
		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{
				// read the current frame
				this.capture.read(frame);
				
				// if the frame is not empty, process it
				if (!frame.empty())
				{
					// face detection
					this.detectAndDisplay(frame);
				}
				
			}
			catch (Exception e)
			{
				// log the (full) error
				System.err.println("Exception during the image elaboration: " + e);
			}
		}
		
		return frame;
	}
	
	/**
	 * Method for face detection and BPM tracking
	 * 
	 * @param frame
	 *            it looks for faces in this frame
	 */
	private void detectAndDisplay(Mat frame)
	{
		MatOfRect faces = new MatOfRect();
		Mat grayFrame = new Mat();
		
		// convert the frame in gray scale
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		// equalize the frame histogram to improve the result
		Imgproc.equalizeHist(grayFrame, grayFrame);
		
		
		// detect faces
		this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 4, 0 | Objdetect.CASCADE_SCALE_IMAGE,
				new Size(100,100), new Size(400,400));
				
		// each rectangle in faces is a face: draw them!
		
		
		Rect[] facesArray = faces.toArray();
		for (int i = 0; i < facesArray.length; i++)
		{

			//Capture Forehead from detected face
			Rect temp = facesArray[i];
			
			temp.x = (int)(temp.x + temp.width * 0.5 - (temp.width * 0.25 / 2.0));
			temp.y = (int)(temp.y + temp.height * 0.18 - (temp.height * 0.15 / 2.0));
			temp.width =(int) (temp.width*0.25);
			temp.height =(int) (temp.height*0.15);
			
			//Detect and ignore outlier faces
			if(start == 1)
			{
				prev_x = temp.x;
				prev_y = temp.y;
				start = 0;
			}
			else
			{
				if(Math.abs(temp.x-prev_x)>30 || Math.abs(temp.y-prev_y)>30)
					continue;
			}
			
			prev_x = temp.x;
			prev_y=temp.y;

			double r_sum=0, g_sum=0 , b_sum =0;
			int count =0;
			for (int j = temp.x; j<temp.x+temp.width;j++)
				for (int k = temp.y; k<temp.y+temp.height;k++)
				{
					 double[] pixel_bgr = frame.get(i,j);

					 b_sum=b_sum+pixel_bgr[0];
					 g_sum=g_sum+pixel_bgr[1];
					 r_sum=r_sum+pixel_bgr[2];
					 count++;

				}
			//double avg = (g_sum/count);
			double avg = ((b_sum/count)+(g_sum/count)+(r_sum/count))/3;
			Timestamp timeStamp = new Timestamp(System.currentTimeMillis());
			Val_Buffer.add(avg);
			long start = 1493715946035L;
			long x = timeStamp.getTime() - start;
			double time = Double.parseDouble(x+"");
			Time_Buffer.add(time);

			if(Val_Buffer.size() == sample_count)
			{
				double bpm = Process_BPM();
				System.out.println("BPM : "+ bpm);
				Val_Buffer.clear();
				Time_Buffer.clear();
			}
						
			//Removed
			Imgproc.rectangle(frame, temp.tl(), temp.br(), new Scalar(0, 255, 0), 2);
		}	
	}
	
	/*
	 * Proccess the Heart Beats per minute from ArrayList Val_Buffer
	 *  
	 */
	protected ArrayList<Double> linspace(double starttime, double endtime, int samplecount) 
	{
		ArrayList<Double> res = new ArrayList<Double>(); 
		double time_diff = Time_Buffer.get(sample_count-1)-Time_Buffer.get(0)+1;
		double incr = time_diff/samplecount;
		
		double time = Time_Buffer.get(0);
		while(res.size()!=sample_count)
		{
			res.add(time);
			time = time+incr;
		}
		
		return res;
	}
	
	
	// Interpolate Data
	public static final double[] interp(double[] time_buf, double[] val_buf, double[] even_times) throws IllegalArgumentException {

        if (time_buf.length != val_buf.length) {
            throw new IllegalArgumentException("X and Y must be the same length");
        }
        if (time_buf.length == 1) {
            throw new IllegalArgumentException("X must contain more than one value");
        }
        double[] dx = new double[time_buf.length - 1];
        double[] dy = new double[time_buf.length - 1];
        double[] slope = new double[time_buf.length - 1];
        double[] intercept = new double[time_buf.length - 1];

        // Calculate the line equation (i.e. slope and intercept) between each point
        for (int i = 0; i < time_buf.length - 1; i++) {
            dx[i] = time_buf[i + 1] - time_buf[i];
            if (dx[i] == 0) {
                throw new IllegalArgumentException("X must be montotonic. A duplicate " + "x-value was found");
            }
            if (dx[i] < 0) {
                throw new IllegalArgumentException("X must be sorted");
            }
            dy[i] = val_buf[i + 1] - val_buf[i];
            slope[i] = dy[i] / dx[i];
            intercept[i] = val_buf[i] - time_buf[i] * slope[i];
        }

        // Perform the interpolation here
        double[] yi = new double[even_times.length];
        for (int i = 0; i < even_times.length; i++) {
            if ((even_times[i] > time_buf[time_buf.length - 1]) || (even_times[i] < time_buf[0])) {
                yi[i] = Double.NaN;
            }
            else {
                int loc = Arrays.binarySearch(time_buf,  even_times[i]);
                if (loc < -1) {
                    loc = -loc - 2;
                    yi[i] = slope[loc] * even_times[i] + intercept[loc];
                }
                else {
                    yi[i] = val_buf[loc];
                }
            }
        }

        return yi;
    }
	
	public static double[] hamming(final int n) {
        double[] window = new double[n];
        for (int i = 0; i < n; i++) {
            window[i] = 0.54 - 0.46 * FastMath.cos(2 * Math.PI * ((double) i / (double) (n - 1)));
        }
        return window;
    }
	
	public static double[] multiply(double[] a, double[] b) {
		double[] res = new double[a.length];
		for (int i = 0; i < a.length; i++) {
			res[i] = a[i] * b[i];
		}
		return res;
	}

	public static double[] submin(double[] interpolated)
	{
		
		double sum = 0;
		for(int i = 0; i<interpolated.length;i++)
			sum = sum+interpolated[i];
		
		double mean = sum/interpolated.length;
		
		
		for(int i = 0; i<interpolated.length;i++)
			interpolated[i] = interpolated[i] - mean;
		
		return interpolated;
		
	}
	
	protected int Process_BPM()
	{
		System.out.println("Processing BPM");
		try
		{
		//Time ArrayList linear spaced
		ArrayList<Double> eventimes = linspace(Time_Buffer.get(0),Time_Buffer.get(sample_count-1),sample_count);

		double[] time_buf = new double[Time_Buffer.size()];
		double[] val_buf = new double[Val_Buffer.size()];
		double[] even_times = new double[eventimes.size()];
		int c = 0;
		for(Object i:Time_Buffer)
		{
			double tmp = (double) i;
			time_buf[c++]= tmp;
		}
		c = 0;
		for(Object i:Val_Buffer)
		{
			double tmp = (double) i;
			val_buf[c++]= tmp;
		}
		c = 0;
		System.out.println("Even Times Data");
		for(Object i:eventimes)
		{
			double tmp = (double) i;
			System.out.println(tmp);
			even_times[c++]= tmp;
		}
		
		double[] interpolated = interp(time_buf,val_buf,even_times);

		System.out.println("Interpolated Data");
		for(double x:interpolated)
			System.out.println(x);
		
		//Hamming
		double[] ham = hamming(sample_count);
		interpolated = multiply(ham,interpolated);
		
		interpolated = submin(interpolated);
		
		FastFourierTransformer trans = new FastFourierTransformer(DftNormalization.STANDARD);
		Complex[] values = trans.transform(interpolated, TransformType.FORWARD);	
		
		double magnitude[] = new double[values.length];
		// calculate power spectrum (magnitude) values from fft[]
		for(int i = 0 ; i < sample_count ;i++)
		{
			Complex value = values[i];
			double re = value.getReal();
			double im = value.getImaginary();
			magnitude[i] = Math.sqrt(re*re+im*im);
			System.out.println(magnitude[i]);
		}
		// find largest peak in power spectrum
		double max_magnitude = Double.NEGATIVE_INFINITY;
		int max_index = -1;
		for (int i = 0 ;i< sample_count;i++)
		{
		  if( magnitude[i] > max_magnitude)
		  {
		    max_magnitude = magnitude[i];
		    max_index = i;
		  }
		}
		// convert index of largest peak to frequency
		double freq = max_index * 100 / sample_count;
		
		return (int) freq;
		}
		catch(Exception e)
		{
			System.out.println(e);
		}	
		return 0;
	}
	
	
	
	
	/**
	 * Method for loading a classifier trained set from disk
	 * 
	 * @param classifierPath
	 *            the path on disk where a classifier trained set is located
	 */
	private void checkboxSelection()
	{
		// load the classifier(s)
		
		this.faceCascade.load("resources/lbpcascades/lbpcascade_frontalface.xml");
		
		// now the video capture can start
		this.cameraButton.setDisable(false);
	}
	
	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition()
	{
		if (this.timer!=null && !this.timer.isShutdown())
		{
			try
			{
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}
		
		if (this.capture.isOpened())
		{
			// release the camera
			this.capture.release();
		}
	}
	
	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image)
	{
		Utils.onFXThread(view.imageProperty(), image);
	}
	
	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed()
	{
		this.stopAcquisition();
	}
	
}
