
/*
* 
        DESCRIPTION:
            This is a program for detecting the color of a card based on its image, and the card color is returned on the console.

        INSTRUCTIONS :
            You need to specify the path to the symbols.jpg file with color patterns in the "templateSuitImage" variable in the main() function.The file is located in the folder.

            Sample card images are located in the folder.

            In the main() function, you can change the.jpg file name in the "sampleImageOriginal" variable to four names
            (exampleH, exampleD, exampleS, exampleC), each representing a card image with a different color.

            If the user wants to use their own card image, the following conditions must be met :
                - It must be an image of a single card.
                - The external contours of the card must be visible, preferably against a dark background.
                - The card color must be located in the upper left corner.

*/





#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

vector<Point> cardCorners;
vector<Point> newCardCorners;

vector<vector<Point>> sampleImageContours;
vector<Vec4i> sampleImageHierarchy;

Mat sampleImage, sampleImageOriginal, croppedCard;
Mat suitImage, suitImageOriginal;
Mat templateSuitImage;
string symbolType;

void getCardContour(vector<vector<Point>> aContours) {

    vector<vector<Point>> contourPolygon(aContours.size());
    int area;
    float polygonParameter;
    int maxContourArea = 0;

    for (int i = 0; i < aContours.size(); i++) {
        area = contourArea(aContours[i]);
        polygonParameter = arcLength(aContours[i], true);
        approxPolyDP(aContours[i], contourPolygon[i], 0.02 * polygonParameter, true);
        if (area > maxContourArea && contourPolygon[i].size() == 4) {
            maxContourArea = area;
            cardCorners = { contourPolygon[i][0], contourPolygon[i][1], contourPolygon[i][2], contourPolygon[i][3] };
        }
    }
}

void orderCardCorners() {

    vector<int> xySumOfEachCorner, xySubOfEachCorner;

    for (int i = 0; i < 4; i++) {
        xySumOfEachCorner.push_back(cardCorners[i].x + cardCorners[i].y);
        xySubOfEachCorner.push_back(cardCorners[i].x - cardCorners[i].y);
    }

    newCardCorners.push_back(cardCorners[min_element(xySumOfEachCorner.begin(), xySumOfEachCorner.end()) - xySumOfEachCorner.begin()]);
    newCardCorners.push_back(cardCorners[max_element(xySubOfEachCorner.begin(), xySubOfEachCorner.end()) - xySubOfEachCorner.begin()]);
    newCardCorners.push_back(cardCorners[min_element(xySubOfEachCorner.begin(), xySubOfEachCorner.end()) - xySubOfEachCorner.begin()]);
    newCardCorners.push_back(cardCorners[max_element(xySumOfEachCorner.begin(), xySumOfEachCorner.end()) - xySumOfEachCorner.begin()]);

}

void cropCard(float height, float width) {

    Point2f src[4] = { newCardCorners[0], newCardCorners[1] ,newCardCorners[2] ,newCardCorners[3] };
    Point2f distance[4] = { {0.0f,0.0f}, {width,0.0f}, {0.0f,height}, {width,height} };

    Mat matrix = getPerspectiveTransform(src, distance);
    warpPerspective(sampleImageOriginal, croppedCard, matrix, Point(width, height));
}

void extractSuitImage() {

    Rect roi(0, 75, 80, 150);
    suitImageOriginal = croppedCard(roi);
    resize(suitImageOriginal, suitImageOriginal, Point(300, 562));
}

vector<Mat> getBoundingRect(vector<vector<Point>> aContours, Mat aImage, string aDescription) {

    vector<vector<Point>> pContourPolygon(aContours.size());
    vector<Rect> pBoundingRect(aContours.size());
    vector<Mat> pBoundingRectImage(aContours.size());

    for (int i = 0; i < aContours.size(); i++) {
        approxPolyDP(aContours[i], pContourPolygon[i], 0, true);
        pBoundingRect[i] = boundingRect(pContourPolygon[i]);
        pBoundingRectImage[i] = aImage(pBoundingRect[i]);
        resize(pBoundingRectImage[i], pBoundingRectImage[i], Size(500, 500));
    }
    return pBoundingRectImage;
}

vector<vector<vector<Point>>> getPolygonApproximationContour(vector < Mat > aImage, string aDescription) {

    vector<vector<vector<Point>>> pImagePolyApproxContours(aImage.size());
    vector<vector<Vec4i>> pImagePolyApproxContoursHierarchy(aImage.size());
    vector<Mat> pBackground(aImage.size());
    float pPolygonParameter;

    for (int i = 0; i < aImage.size(); i++) {
        findContours(aImage[i], pImagePolyApproxContours[i], pImagePolyApproxContoursHierarchy[i], RETR_LIST, CHAIN_APPROX_SIMPLE);
        for (int j = 0; j < pImagePolyApproxContours[i].size(); j++) {
            pPolygonParameter = arcLength(pImagePolyApproxContours[i][j], true);
            approxPolyDP(pImagePolyApproxContours[i][j], pImagePolyApproxContours[i][j], 0.02 * pPolygonParameter, true);
        }
        pBackground[i] = Mat::zeros(aImage[i].size(), CV_8UC3);
        drawContours(pBackground[i], pImagePolyApproxContours[i], -1, Scalar(0, 255, 0), 2);
    }
    return pImagePolyApproxContours;
}

vector<Point> get2ndLargestContour(vector<vector<Point>> aContours) {

    vector<Point> p2ndLargestContour;
    double pMaxArea = -1, p2ndMaxArea = -1;

    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < aContours.size(); i++) {
            double pArea = contourArea(aContours[i]);
            if (j == 0) {
                if (pArea > pMaxArea) {
                    pMaxArea = pArea;
                }
            }
            else {
                if (pArea > p2ndMaxArea && pArea < pMaxArea) {
                    p2ndMaxArea = pArea;
                    p2ndLargestContour = aContours[i];
                }
            }
        }
    }
    return p2ndLargestContour;
}

vector<vector<Point>> getTemplateSuitContours() {

    vector<vector<Point>>pTemplateSuitImageContours;
    vector<Vec4i> pTemplateSuitImageHierarchy;

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    cvtColor(templateSuitImage, templateSuitImage, COLOR_BGR2GRAY);
    GaussianBlur(templateSuitImage, templateSuitImage, Size(3, 3), 3, 0);
    Canny(templateSuitImage, templateSuitImage, 25, 75);
    dilate(templateSuitImage, templateSuitImage, kernel);
    threshold(templateSuitImage, templateSuitImage, 50, 255, THRESH_BINARY);

    findContours(templateSuitImage, pTemplateSuitImageContours, pTemplateSuitImageHierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Mat> pTemplateSuitBoudingRectImage = getBoundingRect(pTemplateSuitImageContours, templateSuitImage, "template suit");

    vector<vector<vector<Point>>> pTemplateSuitPolyApproxContours = getPolygonApproximationContour(pTemplateSuitBoudingRectImage, "template suit");

    vector<vector<Point>> pTemplateSuitFinalContours(pTemplateSuitPolyApproxContours.size());

    vector<Mat> pBackground(pTemplateSuitPolyApproxContours.size());

    for (int i = 0; i < pTemplateSuitPolyApproxContours.size(); i++) {
        pTemplateSuitFinalContours[i] = get2ndLargestContour(pTemplateSuitPolyApproxContours[i]);
        pBackground[i] = Mat::zeros(pTemplateSuitBoudingRectImage[0].size(), CV_8UC3);
        drawContours(pBackground[i], pTemplateSuitFinalContours, i, Scalar(0, 255, 0), 2);
    }
    return pTemplateSuitFinalContours;

}

void sampleImageProcessing() {

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

    cvtColor(sampleImageOriginal, sampleImage, COLOR_BGR2GRAY);
    GaussianBlur(sampleImage, sampleImage, Size(3, 3), 3, 0);
    Canny(sampleImage, sampleImage, 25, 75);
    dilate(sampleImage, sampleImage, kernel);
    threshold(sampleImage, sampleImage, 220, 255, THRESH_BINARY);

    findContours(sampleImage, sampleImageContours, sampleImageHierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    drawContours(sampleImage, sampleImageContours, -1, Scalar(0, 255, 0), 1);

    getCardContour(sampleImageContours);
    orderCardCorners();
    cropCard(560, 400);
    extractSuitImage();
}

vector<vector<Point>> getSampleCardSuitContour() {

    vector<vector<Point>> pSampleSuitImageContours;
    vector<Vec4i> pSampleSuitImageHierarchy;

    sampleImageProcessing();

    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));

    cvtColor(suitImageOriginal, suitImage, COLOR_BGR2GRAY);
    threshold(suitImage, suitImage, 80, 255, THRESH_BINARY);
    GaussianBlur(suitImage, suitImage, Size(3, 3), 3, 0);
    Canny(suitImage, suitImage, 25, 75);
    dilate(suitImage, suitImage, kernel);

    findContours(suitImage, pSampleSuitImageContours, pSampleSuitImageHierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> pSampleCardSuitContour(1);
    pSampleCardSuitContour[0] = get2ndLargestContour(pSampleSuitImageContours);

    vector<Mat> pSampleCardSuitBoudingRectImage = getBoundingRect(pSampleCardSuitContour, suitImage, "Sample card suit");

    vector<vector<vector<Point>>> pSampleCardSuitPolyApproxContours = getPolygonApproximationContour(pSampleCardSuitBoudingRectImage, "Sample card suit");

    vector<vector<Point>> pSampleCardSuitFinalContour(1);
    pSampleCardSuitFinalContour[0] = get2ndLargestContour(pSampleCardSuitPolyApproxContours[0]);

    Mat pBackground = Mat::zeros(pSampleCardSuitBoudingRectImage[0].size(), CV_8UC3);
    drawContours(pBackground, pSampleCardSuitFinalContour, -1, Scalar(0, 255, 0), 2);

    return pSampleCardSuitFinalContour;
}

void identifyCardSuit() {

    double symbolMatchingLowestValue = -1;
    double currentSymbolMatchingLowestValue = -1;

    vector<vector<Point>> pTemplateSuitContours = getTemplateSuitContours();
    vector<vector<Point>> pSampleCardSuitContour = getSampleCardSuitContour();

    for (int i = pTemplateSuitContours.size() - 1; i >= 0; i--) {
        int pCornerNumSampleCardSuit = pSampleCardSuitContour[0].size();
        int pCornerNumTemplateSuit = pTemplateSuitContours[i].size();
        currentSymbolMatchingLowestValue = matchShapes(pSampleCardSuitContour[0], pTemplateSuitContours[i], CONTOURS_MATCH_I1, 0);
        if (symbolMatchingLowestValue < 0 || symbolMatchingLowestValue > currentSymbolMatchingLowestValue && pCornerNumSampleCardSuit > 4 && pCornerNumTemplateSuit > 4) {
            symbolMatchingLowestValue = currentSymbolMatchingLowestValue;
            if (i == 3) {
                symbolType = "Heart";
            }
            if (i == 2) {
                symbolType = "Club";
            }
            if (i == 1) {
                symbolType = "Spade";
            }
        }
        else if (pCornerNumSampleCardSuit <= 4) {
            symbolType = "Diamond";
        }
    }
    cout << "\n\n\n\nCard Suit: " << symbolType << "\n\n\n\n" << endl;
}

void performOperation() {

    identifyCardSuit();

    resize(sampleImageOriginal, sampleImageOriginal, cv::Size(sampleImageOriginal.cols * 0.5, sampleImageOriginal.rows * 0.5));
    imshow("Przykladowa karta", sampleImageOriginal);

    waitKey(0);
}

int main() {

    templateSuitImage = imread("Suit Template/suit_template.jpg");

    sampleImageOriginal = imread("Sample Cards/exampleD.jpg");

    if (sampleImageOriginal.empty() || templateSuitImage.empty())
    {
        std::cout << "Failed to open the image file." << std::endl;
        return -1;
    }

    performOperation();

    return 0;
}
