#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Analysis/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <sstream>
#include <set>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

// === AST Serializer ===
json::Value SerializeAST(const Stmt *stmt);
json::Value SerializeAST(const Decl *decl) {
    json::Object obj;
    obj["kind"] = decl->getDeclKindName();
    if (const auto *fd = dyn_cast<FunctionDecl>(decl)) {
        obj["name"] = fd->getNameAsString();
        if (fd->hasBody())
            obj["body"] = SerializeAST(fd->getBody());
    }
    return std::move(obj);
}

json::Value SerializeAST(const Stmt *stmt) {
    json::Object obj;
    obj["kind"] = stmt->getStmtClassName();
    json::Array children;
    for (const Stmt *child : stmt->children()) {
        if (child)
            children.push_back(SerializeAST(child));
    }
    obj["children"] = std::move(children);
    return std::move(obj);
}

// === CFG Serializer ===
json::Value SerializeCFG(const CFG &cfg) {
    json::Array blocks;
    for (const CFGBlock *block : cfg) {
        json::Object b;
        b["id"] = block->getBlockID();
        json::Array stmts;

        for (const auto &elem : *block) {
            if (Optional<CFGStmt> CS = elem.getAs<CFGStmt>()) {
                const Stmt *stmt = CS->getStmt();
                stmts.push_back(stmt->getStmtClassName());
            }
        }

        b["statements"] = std::move(stmts);
        blocks.push_back(std::move(b));
    }
    return std::move(blocks);
}

// === Visitor ===
class JulietVisitor : public RecursiveASTVisitor<JulietVisitor> {
public:
    explicit JulietVisitor(ASTContext *context) : ctx(context) {}

    bool VisitFunctionDecl(FunctionDecl *FD) {
        if (!FD->hasBody()) return true;

        std::string fname = FD->getNameAsString();
        
        // Skip built-in C functions and compiler-generated functions
        if (isBuiltinOrStandardFunction(fname, FD)) {
            return true;
        }

        int label = fname.find("bad") != std::string::npos ? 1 : 0;

        std::unique_ptr<CFG> cfg = CFG::buildCFG(FD, FD->getBody(), ctx, CFG::BuildOptions());
        if (!cfg) {
            llvm::errs() << "ERROR: Failed to build CFG for function: " << fname << "\n";
            return true;
        }

        json::Object out;
        out["function"] = fname;
        out["label"] = label;
        out["ast"] = SerializeAST(FD);
        out["cfg"] = SerializeCFG(*cfg);

        std::error_code EC;
        llvm::raw_fd_ostream llvm_out("outputs/" + fname + ".json", EC, llvm::sys::fs::OF_Text);
        if (EC) {
            llvm::errs() << "ERROR: Failed to write output file for function " << fname 
                         << ": " << EC.message() << "\n";
            return true;
        }
        
        llvm_out << json::Value(std::move(out));
        
        // Only output on success if there were issues (removed the normal success message)
        return true;
    }

private:
    bool isBuiltinOrStandardFunction(const std::string& fname, FunctionDecl *FD) {
        // Skip functions from system headers
        SourceManager &SM = ctx->getSourceManager();
        if (SM.isInSystemHeader(FD->getLocation())) {
            return true;
        }
        
        // Skip common C standard library functions
        static const std::set<std::string> builtinFunctions = {
            "printf", "fprintf", "sprintf", "snprintf", "scanf", "fscanf", "sscanf",
            "malloc", "calloc", "realloc", "free",
            "strlen", "strcpy", "strncpy", "strcmp", "strncmp", "strcat", "strncat",
            "strchr", "strrchr", "strstr", "strtok",
            "memcpy", "memmove", "memset", "memcmp", "memchr",
            "fopen", "fclose", "fread", "fwrite", "fseek", "ftell", "rewind",
            "exit", "abort", "atexit", "system",
            "atoi", "atol", "atof", "strtol", "strtoul", "strtod",
            "isalpha", "isdigit", "isalnum", "isspace", "toupper", "tolower",
            "time", "clock", "difftime", "mktime", "asctime", "ctime", "gmtime", "localtime",
            "rand", "srand", "abs", "labs", "div", "ldiv",
            "qsort", "bsearch",
            "getenv", "putenv", "setenv", "unsetenv"
        };
        
        if (builtinFunctions.count(fname) > 0) {
            return true;
        }
        
        // Skip functions starting with underscores (typically compiler-generated)
        if (fname.length() > 0 && fname[0] == '_') {
            return true;
        }
        
        // Skip functions with no user-defined body (declarations only)
        if (!FD->hasBody() || !FD->isThisDeclarationADefinition()) {
            return true;
        }
        
        return false;
    }

    ASTContext *ctx;
};

class JulietConsumer : public ASTConsumer {
public:
    explicit JulietConsumer(ASTContext *context) : visitor(context) {}
    void HandleTranslationUnit(ASTContext &ctx) override {
        visitor.TraverseDecl(ctx.getTranslationUnitDecl());
    }
private:
    JulietVisitor visitor;
};

class JulietFrontendAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef) override {
        return std::make_unique<JulietConsumer>(&CI.getASTContext());
    }
};

// === CLI Setup ===
static llvm::cl::OptionCategory ToolCategory("juliet-ast-cfg");

int main(int argc, const char **argv) {
    auto expectedParser = CommonOptionsParser::create(argc, argv, ToolCategory);
    if (!expectedParser) {
        llvm::errs() << expectedParser.takeError();
        return 1;
    }

    ClangTool Tool(expectedParser->getCompilations(), expectedParser->getSourcePathList());
    return Tool.run(newFrontendActionFactory<JulietFrontendAction>().get());
}

