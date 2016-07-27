using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Emgu.CV.Structure;
using Emgu.CV;
using HandGestureRecognition.SkinDetector;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;

namespace HandGestureRecognition
{
    public partial class Form1 : Form
    {
        IColorSkinDetector skinDetector;

        Image<Bgr, Byte> currentFrame;
        Image<Bgr, Byte> currentFrameCopy;
                
        Capture grabber;
        
        int frameWidth;
        int frameHeight;
        
        Hsv hsv_min;
        Hsv hsv_max;
        Ycc YCrCb_min;
        Ycc YCrCb_max;
        
        VectorOfInt hull;
        VectorOfPoint filteredHull;
        Mat convexityDefect;
        Point[] defectArray;
        Rectangle handRect;
        //MCvBox2D box;
        Ellipse ellip;
        bool hold;
        bool open;
        bool move;
        MCvPoint3D32f object_3d;
        PointF object_point;
        PointF detection_point;
        MemStorage storage = new MemStorage();
        VectorOfPoint hand;
        
        public Form1()
        {
            InitializeComponent();

            try
            {
                grabber = new Emgu.CV.Capture();
                //grabber.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.Fps, 25);
                //grabber.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameHeight, 540);
                //grabber.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameWidth, 960);

                grabber.QueryFrame();
                
                frameWidth = grabber.Width;
                frameHeight = grabber.Height;
                hsv_min = new Hsv(0, 45, 0);
                hsv_max = new Hsv(20, 255, 255);
                YCrCb_min = new Ycc(0, 131, 80);
                hold = false;
                open = false;
                move = false;
                object_point = new PointF(frameHeight / 2, frameHeight / 2);
                detection_point = new PointF(0, 0);
                YCrCb_max = new Ycc(255, 185, 135);
                //box = new MCvBox2D();
                ellip = new Ellipse();
                hand = null;
                object_3d = new MCvPoint3D32f(0, 0, 0);

                Application.Idle += new EventHandler(FrameGrabber);
                //grabber.ImageGrabbed += FrameGrabber;
            }
            catch (Exception excpt)
            {
                MessageBox.Show(excpt.Message);
                grabber.Dispose();
            }
        }

        private void FrameGrabber(object sender, EventArgs e)
        {
            try
            {
                Mat frame = new Mat();
                grabber.Retrieve(frame, 0);

                //Mat frame = grabber.QueryFrame();
                if (frame != null)
                {
                    currentFrame = frame.ToImage<Bgr, Byte>();
                }

                if (currentFrame != null)
                {
                    currentFrameCopy = currentFrame.Copy();

                    skinDetector = new YCrCbSkinDetector();

                    Image<Gray, Byte> skin = skinDetector.DetectSkin(currentFrameCopy,YCrCb_min,YCrCb_max);

                    ExtractContourAndHull(skin);

                    //DrawAndComputeFingersNum();

                    imageBoxSkin.Image = skin;
                    imageBoxFrameGrabber.Image = currentFrame;
                    //object_3d = this.ConvertTo3D(this.object_point);
                    //Console.WriteLine("(" + object_3d.x + ", " + object_3d.y + ", " + object_3d.z + ")");
                }
            }
            catch (Exception exp)
            {
                grabber.Dispose();
            }

        }

        private MCvPoint3D32f ConvertTo3D(PointF point)
        {
            float ratioX = (10f/frameWidth);
            float ratioY = (10f*frameHeight/frameWidth);
            return new MCvPoint3D32f(this.object_point.X*ratioX, this.object_point.Y*ratioY, 0);
        }
        
        private PointF MaxYPoint(Point[] arr)
        {
            PointF max = new PointF(0,0);
            float maxLenght = 100000;
            for (int i = 0; i < arr.Length; i++)
            {
                if (arr[i].Y < maxLenght)
                {
                    maxLenght = arr[i].Y;
                    max = new PointF(arr[i].X, arr[i].Y);
                }
            }
            return max;
        }
        
        private void ExtractContourAndHull(Image<Gray, byte> img)
        {
            VectorOfPoint biggestContour = FindLargestContour(img);

            if (biggestContour != null)
            {
                //double Perimeter = CvInvoke.ArcLength(biggestContour, true);
                VectorOfPoint approxContour = null;// new VectorOfPoint();
                CvInvoke.ApproxPolyDP(biggestContour, approxContour, 5, true);//Perimeter*0.0025
                biggestContour = approxContour;
                hand = biggestContour;

                CvInvoke.DrawContours(currentFrame, biggestContour, 0, new MCvScalar(255, 0, 0), 2);
                
                if (move)
                {
                    PointF max = MaxYPoint(hand.ToArray());
                    object_point.X += max.X - detection_point.X;
                    object_point.Y += max.Y - detection_point.Y;
                    currentFrame.Draw(new CircleF(detection_point, 5f), new Bgr(Color.Black), 2);
                }
                currentFrame.Draw(new CircleF(object_point, 5f), new Bgr(Color.Pink), 2);

                // the hull is returned as an integer vector consisting of indices in the original contour (last parameter "bool returnPoints" must be false). 
                CvInvoke.ConvexHull(biggestContour, hull, false, false);

                //currentFrame.DrawPolyline(hull, true, new Bgr(200, 125, 75), 2);

                CvInvoke.ConvexityDefects(biggestContour, hull, convexityDefect);

                //convexity defect is a four channel mat, when k rows and 1 cols, where k = the number of convexity defects. 
                if (!convexityDefect.IsEmpty)
                {
                    //Data from Mat are not directly readable so we convert it to Matrix<>
                    Matrix<int> m = new Matrix<int>(convexityDefect.Rows, convexityDefect.Cols, convexityDefect.NumberOfChannels);
                    convexityDefect.CopyTo(m);

                    for (int i = 0; i < m.Rows; i++)
                    {
                        int startIdx = m.Data[i, 0];
                        int endIdx = m.Data[i, 1];
                        int depthIdx = m.Data[i,2];
                        Point startPoint = biggestContour[startIdx];
                        Point endPoint = biggestContour[endIdx];
                        Point depthPoint = biggestContour[depthIdx];
                        //draw  a line connecting the convexity defect start point and end point in thin red line
                        CvInvoke.Line(currentFrame, startPoint, endPoint, new MCvScalar(0, 0, 255));
                    }
                }
            }
        }
        /*
        private void DrawAndComputeFingersNum()
        {
            int fingerNum = 0;

            #region hull drawing
            //for (int i = 0; i < filteredHull.Total; i++)
            //{
            //    PointF hullPoint = new PointF((float)filteredHull[i].X,
            //                                  (float)filteredHull[i].Y);
            //    CircleF hullCircle = new CircleF(hullPoint, 4);
            //    currentFrame.Draw(hullCircle, new Bgr(Color.Aquamarine), 2);
            //}
            #endregion
            if (convexityDefect == null)
            {
                return;
            }
            #region defects drawing
            for (int i = 0; i < convexityDefect.Rows; i++)
            {
                PointF startPoint = new PointF((float)defectArray[i].StartPoint.X,
                                                (float)defectArray[i].StartPoint.Y);

                PointF depthPoint = new PointF((float)defectArray[i].DepthPoint.X,
                                                (float)defectArray[i].DepthPoint.Y);

                PointF endPoint = new PointF((float)defectArray[i].EndPoint.X,
                                                (float)defectArray[i].EndPoint.Y);

                LineSegment2D startDepthLine = new LineSegment2D(defectArray[i].StartPoint, defectArray[i].DepthPoint);

                LineSegment2D depthEndLine = new LineSegment2D(defectArray[i].DepthPoint, defectArray[i].EndPoint);

                CircleF startCircle = new CircleF(startPoint, 5f);

                CircleF depthCircle = new CircleF(depthPoint, 5f);

                CircleF endCircle = new CircleF(endPoint, 5f);

                //Custom heuristic based on some experiment, double check it before use
                if ((startCircle.Center.Y < box.center.Y || depthCircle.Center.Y < box.center.Y) && (startCircle.Center.Y < depthCircle.Center.Y) && (Math.Sqrt(Math.Pow(startCircle.Center.X - depthCircle.Center.X, 2) + Math.Pow(startCircle.Center.Y - depthCircle.Center.Y, 2)) > box.size.Height / 6.5))
                {
                    fingerNum++;
                    currentFrame.Draw(startDepthLine, new Bgr(Color.Green), 2);
                    //currentFrame.Draw(depthEndLine, new Bgr(Color.Magenta), 2);
                }


                //currentFrame.Draw(startCircle, new Bgr(Color.Red), 2);
                //currentFrame.Draw(depthCircle, new Bgr(Color.Yellow), 5);
                //currentFrame.Draw(endCircle, new Bgr(Color.DarkBlue), 4);
            }
            if (fingerNum == 5)
            {
                open = true;
                hold = false;
                move = false;
            }
            if (open && fingerNum < 2)
            {
                hold = true;
                open = false;
            }
            //Console.WriteLine(hold);
            if (hold && hand != null )//&& hand.InContour(object_point) > 0
            {
                move = true;
            }
            if (move)
            {
                detection_point = MaxYPoint(hand.ToArray());
            }
            //Console.WriteLine(detection_point);

            #endregion

            //MCvFont font = new MCvFont(Emgu.CV.CvEnum.FONT.CV_FONT_HERSHEY_DUPLEX, 5d, 5d);
            //currentFrame.Draw(fingerNum.ToString(), ref font, new Point(50, 150), new Bgr(Color.White));
        }
        */
        public static VectorOfPoint FindLargestContour(IInputOutputArray img)
        {
            int largest_contour_index = -1;
            double largest_area = 0;
            VectorOfPoint largestContour = null;

            using (Mat hierachy = new Mat())
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                IOutputArray hirarchy;

                CvInvoke.FindContours(img, contours, hierachy, RetrType.List, ChainApproxMethod.ChainApproxSimple);
                Console.WriteLine("contours={0}", contours.Size);

                for (int i = 0; i < contours.Size; i++)
                {
                    MCvScalar color = new MCvScalar(0, 0, 255);

                    double a = CvInvoke.ContourArea(contours[i], false);  //  Find the area of contour
                    if (a > largest_area)
                    {
                        largest_area = a;
                        largest_contour_index = i;                //Store the index of largest contour
                    }

                    //CvInvoke.DrawContours(result, contours, largest_contour_index, new MCvScalar(255, 0, 0));
                }

                if (largest_contour_index >= 0)
                {
                    //CvInvoke.DrawContours(result, contours, largest_contour_index, new MCvScalar(0, 0, 255), 3, LineType.EightConnected, hierachy);
                    largestContour = new VectorOfPoint(contours[largest_contour_index].ToArray());
                }
            }

            return largestContour;
        }


    }
}