#include <opencv2/opencv.hpp>

// 11  8  75
void drawChessBoard(int blocks_per_row, int blocks_per_col, int block_size)
{
    //blocks_per_row=11 //每行11个格子,也就是10个点
    //blocks_per_col=8  //每列8个格子,也就是7个点
    //block_size=75     //每个格子的像素大小
    cv::Size board_size = cv::Size(block_size * blocks_per_row, block_size * blocks_per_col);
    cv::Mat chessboard = cv::Mat(board_size, CV_8UC1);
    unsigned char color = 0;
    for (int i = 0; i < blocks_per_row; i++)
    {
        color = ~color;
        for (int j = 0; j < blocks_per_col; j++)
        {
            chessboard(cv::Rect(i * block_size, j * block_size, block_size, block_size)).setTo(color);
            color = ~color;
        }
    }
    cv::Mat chess_board = cv::Mat(board_size.height + 100, board_size.width + 100, CV_8UC1, cv::Scalar::all(256)); //上下左右留出50个像素空白
    chessboard.copyTo(chess_board.rowRange(50, 50 + board_size.height).colRange(50, 50 + board_size.width));
    cv::imshow("chess_board", chess_board);
    cv::imwrite("chess_board.png", chess_board);
    cv::waitKey(-1);
    cv::destroyAllWindows();
}
int main(){
    int blocks_per_row=11;
    int blocks_per_col=8;
    int block_size=75;
    drawChessBoard(blocks_per_row,blocks_per_col,block_size);
}

// g++ -o drawChessBoard drawChessBoard.cpp `pkg-config --cflags --libs opencv4`