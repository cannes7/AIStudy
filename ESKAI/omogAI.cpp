#include <unistd.h> // For usleep function
#include <iostream>
#include <cstring>
#include <algorithm>
#include <deque>
#define endl '\n'
using namespace std;

// Game board and weight board
int map[19][19];
int w_board[19][19];

// Direction vectors (8 directions)
int dir[8][2] = { { 1,0 },{ 0,1 },{ 1,1 },{ 1,-1 },{ -1,0 },{ 0,-1 },{ -1,-1 },{ -1,1 } };

int n = 19; // Size of the game board

// Weight values (varies by type)
int w[2][6] = { { 0,1,50,9999,500000,10000000 },{ 0,1,12,250,400000,10000000 } };
int w2[2][6][3][2];

int stx, sty; // Starting coordinates for AI search
int ansx, ansy; // Final coordinates found by AI
int real_high[6]; // Highest weights found

// Coordinate structure
typedef struct xy {
    int x, y;
} xy;

// Stone information structure
typedef struct info {
    int num = 0, enemy = 0, emptyspace = 0;
} info;

// Weight information structure
typedef struct info2 {
    int x, y, weight;
};

// Sorting function for descending order by weight
bool cmp(info2 a, info2 b) {
    return a.weight > b.weight;
}

void Print(); // Function to print the game board
void add_weight(int color[2]); // Function to add weights
void search(int cnt, int color); // Search function
void AI(int user_color, int ai_color); // AI function
void input(int type); // Function for user input
void game_type(int type); // Game type function
bool check(int color); // Function to check win conditions

void init() {
    // Initialize weights
    memset(w2, 0, sizeof(w2));
    w2[0][1][0][0] = 2; w2[1][1][0][0] = 1;
    w2[0][1][0][1] = 2; w2[1][1][0][1] = 0;
    w2[0][2][0][0] = 25, w2[1][2][0][0] = 4;
    w2[0][2][0][1] = 25, w2[1][2][0][1] = 1;
    w2[0][2][1][1] = 2; w2[1][2][1][1] = 1;
    w2[0][2][1][0] = 2; w2[1][2][1][0] = 1;
    w2[0][3][0][0] = 521, w2[1][3][0][0] = 105;
    w2[0][3][0][1] = 301; w2[1][3][0][1] = 13;
    w2[0][3][1][0] = 301, w2[1][3][1][0] = 13;
    w2[0][3][1][1] = 301, w2[1][3][1][1] = 13;
    w2[0][4][0][0] = 21000; w2[0][4][1][0] = 20010; w2[0][4][2][0] = 20010;
    w2[1][4][0][0] = 4001; w2[1][4][1][0] = 4001; w2[1][4][2][0] = 4001;
}

void add_weight(int color[2]) {
    // Initialize weight board
    memset(w_board, 0, sizeof(w_board));

    // Assign weights around the stone placements
    for (int type = 0; type < 2; type++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = 0;
                info Count[5]; // Counts of stones in vertical, horizontal, diagonal (right), diagonal (left)

                if (map[i][j]) continue;

                for (int d = 0; d < 4; d++) {
                    int nx, ny;
                    int cnt = 1;
                    int zerocnt = 0; // Up to 2 empty spaces are allowed
                    int zerocnt1 = 0;
                    int remember = 0;
                    int zerocnt2 = 0;
                    int num = 0;
                    int enemy_cnt = 0;
                    int before;

                    while (true) {
                        nx = i + (cnt * dir[d][0]), ny = j + (cnt * dir[d][1]);
                        before = map[nx - dir[d][0]][ny - dir[d][1]];

                        if (nx < 0 || ny < 0 || nx >= n || ny >= n) {
                            if (remember || zerocnt1 == 0) enemy_cnt++;
                            if (before != 0) remember = zerocnt1;
                            break;
                        }

                        if (map[nx][ny] == color[(type + 1) % 2]) {
                            if (remember || zerocnt1 == 0) enemy_cnt++;
                            if (before != 0) remember = zerocnt1;
                            break;
                        }

                        if (map[nx][ny] == color[type]) remember = zerocnt1;
                        if (map[nx][ny] == 0) zerocnt1++;
                        if (zerocnt1 >= 2) break;
                        cnt++;
                    }

                    zerocnt1 = remember;
                    cnt = 1;
                    remember = 0;

                    while (true) {
                        nx = i + (cnt * dir[d + 4][0]), ny = j + (cnt * dir[d + 4][1]);
                        if (nx < 0 || ny < 0 || nx >= n || ny >= n) {
                            if (remember || zerocnt2 == 0) enemy_cnt++;
                            if (before != 0) remember = zerocnt2;
                            break;
                        }

                        if (map[nx][ny] == color[(type + 1) % 2]) {
                            if (remember || zerocnt2 == 0) enemy_cnt++;
                            if (before != 0) remember = zerocnt2;
                            break;
                        }

                        if (map[nx][ny] == color[type]) remember = zerocnt2;
                        if (map[nx][ny] == 0) zerocnt2++;
                        if (zerocnt2 >= 2) break;
                        cnt++;
                    }

                    zerocnt2 = remember;
                    zerocnt = zerocnt1 + zerocnt2;
                    Count[d] = { num, enemy_cnt, zerocnt };
                }

                // Reduce weight if opponent blocks my stones
                for (int d = 0; d < 4; d++) {
                    int num = Count[d].num, enemy = Count[d].enemy, emptyspace = Count[d].emptyspace;
                    int temp_w = w2[(type + 1) % 2][num][enemy][emptyspace];

                    if (emptyspace >= 2 || num + emptyspace >= 5) continue;
                    if (num != 4 && enemy >= 2) continue;
                    sum += temp_w;
                }

                w_board[i][j] += sum;
                if (map[i][j]) w_board[i][j] = 0; // Reset weight if there is already a stone
            }
        }
    }
}

bool tf;

// The search function explores possible moves and determines the optimal move.
void search(int cnt, int color) {
    int ncolor[2] = { 0, };  // Array to store colors, representing opponent's and current player's color.

    // Set array based on current player's color.
    if (color == 1) {
        ncolor[0] = 2; ncolor[1] = 1;  // Opponent is 2 (white), current player is 1 (black).
    } else {
        ncolor[0] = 1; ncolor[1] = 2;  // Opponent is 1 (black), current player is 2 (white).
    }

    int high = 0;
    add_weight(ncolor);  // Update weight based on surrounding stones.

    deque <info2> save_pos;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int wow = w_board[i][j];
            if (wow) {
                if (wow == 301 || wow == 302) wow = 24;
                else if (wow >= 118 && wow <= 200) wow = 320;
                save_pos.push_back({ i,j,wow });
                high = max(high, wow);
            }
        }
    }

    sort(save_pos.begin(), save_pos.end(), cmp);

    // Trim positions with weight significantly lower than the highest weight.
    int MAX = save_pos[0].weight;
    int idx = 0;
    for (int i = 1; i < save_pos.size(); i++) {
        idx = i;
        int num = save_pos[i].weight;
        if (num != MAX) break;
    }

    save_pos.erase(save_pos.begin() + idx, save_pos.end());

    int temp_color;
    if (color == 1) temp_color = 2;
    else temp_color = 1;

    // If the opponent has a winning move, return early.
    if (cnt % 2 == 1 && check(temp_color)) {
        return;
    }

    // If the best move is found and conditions are met, set the result and return.
    if (!tf && (cnt % 2 == 1 && ((MAX >= 326 && MAX < 406) || MAX >= 521))) {
        if (!((105 <= MAX && MAX <= 300) || (4000 <= MAX && MAX < 20000))) {
            tf = true;
            ansx = stx, ansy = sty;
            return;
        }
    }

    // Limit search depth to 30 moves.
    if (cnt == 30) {
        return;
    }

    // Recursive search for possible moves.
    if (color == 1) {
        for (int i = 0; i < save_pos.size(); i++) {
            int x = save_pos[i].x, y = save_pos[i].y;
            map[x][y] = color;
            search(cnt + 1, 2);
            map[x][y] = 0;
        }
    } else if (color == 2) {
        for (int i = 0; i < save_pos.size(); i++) {
            int x = save_pos[i].x, y = save_pos[i].y;
            map[x][y] = color;
            search(cnt + 1, 1);
            map[x][y] = 0;
        }
    }
}

void AI(int user_color, int ai_color) {
    tf = false;
    int color[2] = { user_color, ai_color };
    memset(real_high, 0, sizeof(real_high));

    // Setting up AI algorithm
    add_weight(color);  // Update weights based on surrounding stones
    deque<info2> save_pos;
    save_pos.clear();
    int high = 0;

    // Collecting non-zero weights and determining the highest weight position
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int wow = w_board[i][j];
            if (wow) {
                if (wow == 301 || wow == 302) wow = 24;
                else if (wow >= 118 && wow <= 200) wow = 320;
                save_pos.push_back({ i, j, wow });
                if (high < wow) {
                    high = wow;
                    ansx = i, ansy = j;
                }
            }
        }
    }

    sort(save_pos.begin(), save_pos.end(), cmp);

    // Algorithm logic
    int MAX = save_pos[0].weight;

    // If MAX is not in a critical range, perform a limited search
    if (!((MAX >= 326 && MAX < 406) || MAX >= 521)) {
        // Starting search for non-zero weights
        for (int i = 0; i < save_pos.size(); i++) {
            int x = save_pos[i].x, y = save_pos[i].y;
            stx = x, sty = y;
            map[x][y] = ai_color;  // Place AI move
            search(0, user_color);  // Initiate search from this position
            map[x][y] = 0;  // Undo move
        }
    }

    // Apply AI's optimal move
    map[ansx][ansy] = ai_color;
    cout << "ai input ( " << ansx << " , " << ansy << " )" << endl;
}

void Print() {
    cout << "x|y";
    for (int j = 0; j < n; j++) {
        cout.width(3);
        cout << j;
    }
    cout << endl;
    for (int i = 0; i < n; i++) {
        cout.width(3);
        cout << i;
        for (int j = 0; j < n; j++) {
            cout.width(3);
            if (map[i][j])
                cout << map[i][j];
            else
                cout << "";
        }
        cout << endl;
    }
}

void input(int type) {
    int x, y;
    while (true) {
        cout << "What is your next position (x,y)?: ";
        cin >> x >> y;
        if (x >= 0 && y >= 0 && x < n && y < n && map[x][y] == 0) {
            map[x][y] = type;
            break;
        }
        else {
            cout << "Invalid position. Please try another position." << endl;
        }
    }
    cout << endl;
}

bool check(int color) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (map[i][j] == color) {
                for (int d = 0; d < 8; d++) {
                    int cnt = 1;
                    while (true) {
                        int nx = i + (cnt * dir[d][0]), ny = j + (cnt * dir[d][1]);
                        if (nx < 0 || ny < 0 || nx >= n || ny >= n) break;
                        if (map[nx][ny] != color) break;
                        cnt++;
                    }
                    if (cnt == 5) return true;
                }
            }
        }
    }
    return false;
}

void game_type(int type) {
    if (type == 1) {
        cout << "Your color is black(1). Please input 1!" << endl << endl;
        int turn = 0;
        bool victory;
        while (true) {
            // Your turn
            cout << "Your turn" << endl;
            Print();
            input(type);

            tf = check(1);
            if (tf) {
                Print();
                cout << "You win!" << endl;
                return;
            }

            // AI's turn
            cout << "AI's turn" << endl;
            Print();
            usleep(1000 * 1000); // 1000 milliseconds = 1 second
            AI(1, 2);
            tf = check(2);
            if (tf) {
                Print();
                cout << "AI wins!" << endl;
                return;
            }
            turn++;
        }
    }

    else if (type == 2) {
        cout << "Your color is white(2). Please input 2!" << endl << endl;
        int turn = 0;
        while (true) {
            // AI's turn
            cout << "AI's turn" << endl;
            Print();
            usleep(1000 * 1000); // 1000 milliseconds = 1 second
            if (turn == 0) {
                map[9][9] = 1; // Initial move by AI
            }
            else {
                AI(2, 1);
            }

            tf = check(1);
            if (tf) {
                Print();
                cout << "AI wins!" << endl;
                return;
            }

            // Your turn
            cout << "Your turn" << endl;
            Print();
            input(type);

            tf = check(2);
            if (tf) {
                Print();
                cout << "You win!" << endl;
                return;
            }
            turn++;
        }
    }

    else if (type == 3) {
        cout << "Showing AI match!" << endl;
        int turn = 0;
        while (true) {
            // AI 1's turn
            cout << "AI_1's turn" << endl;
            Print();
            usleep(2000 * 1000); // 2000 milliseconds = 2 seconds
            if (turn == 0) {
                map[9][9] = 1; // Initial move by AI 1
            }
            else {
                AI(2, 1);
            }

            tf = check(1);
            if (tf) {
                Print();
                cout << "AI_1 wins!" << endl;
                return;
            }

            // AI 2's turn
            cout << "AI_2's turn" << endl;
            Print();
            usleep(2000 * 1000); // 2000 milliseconds = 2 seconds
            AI(1, 2);
            tf = check(2);
            if (tf) {
                Print();
                cout << "AI_2 wins!" << endl;
                return;
            }
            turn++;
        }
    }
}

int main() {
    init(); // Initialize weights
    cout << "Welcome to Five Stones!" << endl;
    cout << "1: Play as black (1), 2: Play as white (2), 3: Watch AI's match" << endl;
    int start;
    cin >> start;
    game_type(start);
}

