using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Emgu.CV.Structure;
using Emgu.CV;
using System.Drawing;

namespace HandGestureRecognition.SkinDetector
{
    public class YCrCbSkinDetector:IColorSkinDetector
    {
        public override Image<Gray, byte> DetectSkin(Image<Bgr, byte> Img, IColor min, IColor max)
        {
            Image<Ycc, Byte> currentYCrCbFrame = Img.Convert<Ycc, Byte>();
            Image<Gray, byte> skin = new Image<Gray, byte>(Img.Width, Img.Height);
            skin = currentYCrCbFrame.InRange((Ycc)min,(Ycc) max);

            Mat rect_12 = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(12, 12), new Point(6, 6));
            skin = skin.MorphologyEx(Emgu.CV.CvEnum.MorphOp.Erode, rect_12, new Point(-1, -1), 1, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar());

            Mat rect_6 = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(6, 6), new Point(3, 3));
            skin = skin.MorphologyEx(Emgu.CV.CvEnum.MorphOp.Dilate, rect_6, new Point(-1, -1), 2, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar());

            return skin;
        }
        
    }
}
