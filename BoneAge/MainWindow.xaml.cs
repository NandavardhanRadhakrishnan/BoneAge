using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using BoneAge;

namespace FrontEnd
{
    public partial class MainWindow : Window
    {
        System.Uri imagePath;
        bool male = true;
        private void btnOpenImage_Click(object sender, RoutedEventArgs e)
        {
            // Open a file dialog to select an image file
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image files (*.jpg, *.jpeg, *.png) | *.jpg; *.jpeg; *.png";
            if (openFileDialog.ShowDialog() == true)
            {
                // Load the selected image file into an image control
                imagePath = new Uri(openFileDialog.FileName);
                Image image = new Image();
                image.Source = new BitmapImage(imagePath);
                ImagePreview.Source = image.Source;
            }
        }

        private void radioMale_Checked(object sender, RoutedEventArgs e)
        {
            //resultLbl.Content = imagePath.ToString();   
            male = true;

        }

        private void radioFemale_Checked(object sender, RoutedEventArgs e)
        {
            male = false;
        }

        private void btnSubmit_Click(object sender, RoutedEventArgs e)
        {
            if (imagePath == null)
            {
                resultLbl.Content = "please select a file before submitting";
            }
            else
            {
                Console.WriteLine(imagePath);
                resultLbl.Content = imagePath.ToString();
                //var imageBytes = File.ReadAllBytes("file:///C:/Users/prkna/Desktop/bone-age/data/Training/Male/10/3919.png");
                var imageBytes = File.ReadAllBytes(imagePath.ToString().Substring(10));
                if (male)
                {
                    MaleBoneAgeAI.ModelInput data = new MaleBoneAgeAI.ModelInput() { ImageSource = imageBytes };
                    var predictionResult = MaleBoneAgeAI.Predict(data);
                    resultLbl.Content = "Age in months:"+predictionResult.PredictedLabel.ToString();
                }
                else
                {
                    FemaleBoneAgeAI.ModelInput data = new FemaleBoneAgeAI.ModelInput() { ImageSource = imageBytes };
                    var predictionResult = FemaleBoneAgeAI.Predict(data);
                    resultLbl.Content = "Age in months:" + predictionResult.PredictedLabel.ToString();
                }
            }
        }

        public MainWindow()
        {
            InitializeComponent();
        }


    }
}
