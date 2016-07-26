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

namespace HandGestureRecognition
{
    public partial class Form1 : Form
    {

        IColorSkinDetector skinDetector;

        Image<Bgr, Byte> currentFrame;
        Image<Bgr, Byte> currentFrameCopy;
                
        Capture grabber;
        AdaptiveSkinDetector detector;
        
        int frameWidth;
        int frameHeight;
        
        Hsv hsv_min;
        Hsv hsv_max;
        Ycc YCrCb_min;
        Ycc YCrCb_max;
        
        Seq<Point> hull;
        Seq<Point> filteredHull;
        Seq<MCvConvexityDefect> defects;
        MCvConvexityDefect[] defectArray;
        Rectangle handRect;
        MCvBox2D box;
        Ellipse ellip;
        bool hold;
        bool open;
        bool move;
        MCvPoint3D32f object_3d;
        PointF object_point;
        PointF detection_point;
        MemStorage storage = new MemStorage();
        Contour<Point> hand;
        bool transational;
        bool scale;
        float scalefactor;
        bool scaling;
        float radi;

        float ratioX = 1f;//(1000f / 10);
        float ratioY = 1f;//(800f / 2);
        public Form1()
        {
            InitializeComponent();
            grabber = new Emgu.CV.Capture(0);            
            grabber.QueryFrame();
            frameWidth = grabber.Width;
            frameHeight = grabber.Height;            
            detector = new AdaptiveSkinDetector(1, AdaptiveSkinDetector.MorphingMethod.NONE);
            hsv_min = new Hsv(0, 45, 0); 
            hsv_max = new Hsv(20, 255, 255);            
            YCrCb_min = new Ycc(0, 131, 80);
            hold = false;
            open = false;
            move = false;
            object_point = new PointF(frameHeight / 2, frameHeight / 2);
            detection_point = new PointF(0,0);
            YCrCb_max = new Ycc(255, 185, 135);
            box = new MCvBox2D();
            ellip = new Ellipse();
            hand = null;
            scalefactor = 0.2f;
            scale = true;
            scaling = false;
            transational = false;
            radi = 5f;
            object_3d = new MCvPoint3D32f(0, 0, 0);
            Application.Idle += new EventHandler(FrameGrabber);                        
        }

        void FrameGrabber(object sender, EventArgs e)
        {
            currentFrame = grabber.QueryFrame();
            if (currentFrame != null)
            {
                currentFrameCopy = currentFrame.Copy();
                // Uncomment if using opencv adaptive skin detector
                Image<Gray,Byte> skin = new Image<Gray,byte>(currentFrameCopy.Width,currentFrameCopy.Height);                
                detector.Process(currentFrameCopy, skin);                

                //skinDetector = new YCrCbSkinDetector();
                
                //Image<Gray, Byte> skin = skinDetector.DetectSkin(currentFrameCopy,YCrCb_min,YCrCb_max);

                ExtractContourAndHull(skin);
                                
                DrawAndComputeFingersNum();

                imageBoxSkin.Image = skin;
                imageBoxFrameGrabber.Image = currentFrame;
                object_3d = this.ConvertTo3D(this.object_point);
               // Console.WriteLine("(" + object_3d.x + ", " + object_3d.y + ", " + object_3d.z + ")");
            }
        }

        private MCvPoint3D32f ConvertTo3D(PointF point)
        {
            return new MCvPoint3D32f(this.object_point.X*ratioX, this.object_point.Y*ratioY, 0);
        }
        
        private PointF[] MaxYPoints(Point[] arr, int num)
        {
            HashSet<int> set = new HashSet<int>();
            PointF[] result= new PointF[num];
            for (int j = 0; j < num; j++)
            {
                PointF max = new PointF(0, 0);
                float maxLenght = 100000;
                int index = -1;
                for (int i = 0; i < arr.Length; i++)
                {
                    if (arr[i].Y < maxLenght && !set.Contains(i))
                    {
                        maxLenght = arr[i].Y;
                        max = new PointF(arr[i].X, arr[i].Y);
                        index = i;
                    }
                }
                set.Add(index);
                result[j] = max;
            }
                return result;
        }


        private Contour<Point> findContour(Contour<Point> contours)
        {

            Contour<Point> biggestContour = null;
            if (!open || scale)
            {
                Double Result1 = 0;
                Double Result2 = 0;

                while (contours != null)
                {
                    Result1 = contours.Area;
                    if (Result1 > Result2)
                    {
                        Result2 = Result1;
                        biggestContour = contours;
                    }
                    contours = contours.HNext;
                }
                return biggestContour;
            }
            else
            {
                double max = 0;
                while (contours != null)
                {
                    Seq<MCvConvexityDefect> defs = contours.GetConvexityDefacts(storage, Emgu.CV.CvEnum.ORIENTATION.CV_CLOCKWISE);
                    MCvConvexityDefect[] defsArray = defs.ToArray();
                    double current = 0.0;
                    for (int i = 0; i < defsArray.Length; i++)
                    {

                        current += (new LineSegment2D(defsArray[i].StartPoint, defsArray[i].DepthPoint)).Length;
                    }
                    if (current > max)
                    {
                        max = current;
                        biggestContour = contours;
                    }
                    contours = contours.HNext;
                }
                return biggestContour;
            }
        }
        private void ExtractContourAndHull(Image<Gray, byte> skin)
        {
            Contour<Point> contoursMain = skin.FindContours(Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_LIST, storage);
            Contour<Point> contours = contoursMain;
            Contour<Point> biggestContour = findContour(contours);


            if (biggestContour != null)
            {
                //currentFrame.Draw(biggestContour, new Bgr(Color.DarkViolet), 2);
                Contour<Point> currentContour = biggestContour.ApproxPoly(biggestContour.Perimeter * 0.0025, storage);
                currentFrame.Draw(currentContour, new Bgr(Color.LimeGreen), 2);
                biggestContour = currentContour;
                if (transational)
                {
                    hand = biggestContour;
                    if (move)
                    {
                        PointF max = MaxYPoints(hand.ToArray(), 1)[0];
                        object_point.X += max.X - detection_point.X;
                        object_point.Y += max.Y - detection_point.Y;
                        Console.WriteLine("deltaX: " + (max.X - detection_point.X)*ratioX);
                        Console.WriteLine("deltaY: " + (max.Y - detection_point.Y)*ratioY);
                        Console.WriteLine("X: " + object_point.X * ratioX);
                        Console.WriteLine("Y: " + object_point.Y * ratioY);
                        //currentFrame.Draw(new CircleF(detection_point, 5f), new Bgr(Color.Black), 2);
                    }
                    if (object_point != null)
                    {
                        currentFrame.Draw(new CircleF(object_point, 5f), new Bgr(Color.Pink), 2);
                    }
                }
                if (scale)
                {
                    Double R1 = 0;
                    Double R2 = 0;
                    Contour<Point> c = contoursMain;
                    Contour<Point> secondBiggestContour = null;
                    float scale = 1f;
                    hand = biggestContour;
                    Console.WriteLine(hand.Area);
                    while (c != null)
                    {
                        if (hand.Area - c.Area > 700f)
                        {
                            R1 = c.Area;
                        }
                        if (R1 > R2 && hand.Area - c.Area > 700f)
                        {
                            R2 = R1;
                            secondBiggestContour = c;
                        }
                        c = c.HNext;
                    }
                    Contour<Point> hand2 = secondBiggestContour;
                    if (hand2 != null)
                    {
                        currentFrame.Draw(hand2, new Bgr(Color.Blue), 2);
                        currentFrame.Draw(hand, new Bgr(Color.LimeGreen), 2);
                        if (scaling)
                        {
                            hand = biggestContour;
                            PointF max = MaxYPoints(hand.ToArray(), 1)[0];
                            PointF max2 = MaxYPoints(hand2.ToArray(), 1)[0];
                            currentFrame.Draw(new CircleF(max, 5f), new Bgr(Color.LimeGreen), 2);
                            currentFrame.Draw(new CircleF(max2, 5f), new Bgr(Color.Blue), 2);

                            scale = GetScale(max, max2);
                        }
                        radi = scale;
                        Console.WriteLine(radi);
                        Console.WriteLine(scaling);
                    }
                    currentFrame.Draw(new CircleF(object_point, radi), new Bgr(Color.Pink), 2);
                }
                hull = biggestContour.GetConvexHull(Emgu.CV.CvEnum.ORIENTATION.CV_CLOCKWISE);
                box = biggestContour.GetMinAreaRect();
                PointF[] points = box.GetVertices();
                handRect = box.MinAreaRect();
                //currentFrame.Draw(handRect, new Bgr(200, 0, 0), 1);
                Point[] ps = new Point[points.Length];
                for (int i = 0; i < points.Length; i++)
                    ps[i] = new Point((int)points[i].X, (int)points[i].Y);

                //currentFrame.DrawPolyline(hull.ToArray(), true, new Bgr(200, 125, 75), 2);
                //currentFrame.Draw(new CircleF(new PointF(box.center.X, box.center.Y), 3), new Bgr(200, 125, 75), 2);

                //ellip.MCvBox2D= CvInvoke.cvFitEllipse2(biggestContour.Ptr);
                //currentFrame.Draw(new Ellipse(ellip.MCvBox2D), new Bgr(Color.LavenderBlush), 3);

                PointF center;
                float radius;
                //CvInvoke.cvMinEnclosingCircle(biggestContour.Ptr, out  center, out  radius);
                //currentFrame.Draw(new CircleF(center, radius), new Bgr(Color.Gold), 2);

                //currentFrame.Draw(new CircleF(new PointF(ellip.MCvBox2D.center.X, ellip.MCvBox2D.center.Y), 3), new Bgr(100, 25, 55), 2);
                //currentFrame.Draw(ellip, new Bgr(Color.DeepPink), 2);

                //CvInvoke.cvEllipse(currentFrame, new Point((int)ellip.MCvBox2D.center.X, (int)ellip.MCvBox2D.center.Y), new System.Drawing.Size((int)ellip.MCvBox2D.size.Width, (int)ellip.MCvBox2D.size.Height), ellip.MCvBox2D.angle, 0, 360, new MCvScalar(120, 233, 88), 1, Emgu.CV.CvEnum.LINE_TYPE.EIGHT_CONNECTED, 0);
                //currentFrame.Draw(new Ellipse(new PointF(box.center.X, box.center.Y), new SizeF(box.size.Height, box.size.Width), box.angle), new Bgr(0, 0, 0), 2);


                filteredHull = new Seq<Point>(storage);
                for (int i = 0; i < hull.Total; i++)
                {
                    if (Math.Sqrt(Math.Pow(hull[i].X - hull[i + 1].X, 2) + Math.Pow(hull[i].Y - hull[i + 1].Y, 2)) > box.size.Width / 10)
                    {
                        filteredHull.Push(hull[i]);
                    }
                }

                defects = biggestContour.GetConvexityDefacts(storage, Emgu.CV.CvEnum.ORIENTATION.CV_CLOCKWISE);

                defectArray = defects.ToArray();

            }
        }

        private float GetScale(PointF pointF1, PointF pointF2)
        {
            double dX = pointF1.X - pointF2.X;
            double dY = pointF1.Y - pointF2.Y;
            double multi = dX * dX + dY * dY;
            double rad = Math.Round(Math.Sqrt(multi), 3);
            float radius  = Convert.ToSingle(rad);
            Console.WriteLine("Diff: " + radius);
            return radius * this.scalefactor;

        }

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
            if (defects == null)
            {
                return;
            }
            #region defects drawing
            for (int i = 0; i < defects.Total; i++)
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
                    currentFrame.Draw(depthEndLine, new Bgr(Color.Magenta), 2);
                }


                currentFrame.Draw(startCircle, new Bgr(Color.Red), 2);
                currentFrame.Draw(depthCircle, new Bgr(Color.Yellow), 5);
                //currentFrame.Draw(endCircle, new Bgr(Color.DarkBlue), 4);
            }
            if (transational)
            {
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
                if (hold && hand != null && hand.InContour(object_point) > 0)
                {
                    move = true;
                }
                if (move)
                {
                    detection_point = MaxYPoints(hand.ToArray(), 1)[0];
                }
                //Console.WriteLine(detection_point);
            }
            if (scale)
            {
                //if (fingerNum == 5)
                //{
                //    open = true;
                //    hold = false;
                //    scaling = false;
                //}
                //if (open && fingerNum < 2)
                //{
                //    hold = true;
                //    open = false;
                //}
                //if (hold && hand != null && hand.InContour(object_point) > 0)
                //{
                //    scaling = true;
                //}
                //if (fingerNum <= 3)
               // {
                    scaling = true;
               // } else
               // {
               //     scaling = false;
               // }
            }

            #endregion

            MCvFont font = new MCvFont(Emgu.CV.CvEnum.FONT.CV_FONT_HERSHEY_DUPLEX, 5d, 5d);
            currentFrame.Draw(fingerNum.ToString(), ref font, new Point(50, 150), new Bgr(Color.White));
        }
                                      
    }
}