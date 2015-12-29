#include <base/debug/Logger.h>
#include <iostream>
#include <base/time/Global_Timer.h>

using namespace pi;
#define HAS_BOOST
#ifdef HAS_BOOST
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace keywords = boost::log::keywords;
void InitLog() {
  boost::log::register_simple_formatter_factory< boost::log::trivial::severity_level, char >("Severity");
  logging::add_file_log(
   keywords::file_name = "sign_%Y-%m-%d_%H-%M-%S.%N.log",
   keywords::rotation_size = 10 * 1024 * 1024,
   keywords::time_based_rotation = sinks::file::rotation_at_time_point(0, 0, 0),
   keywords::format = "[%TimeStamp%] (%Severity%) : %Message%",
   keywords::min_free_space=3 * 1024 * 1024
   );
  logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::debug);
}

int main()
{
    logging::add_common_attributes();
    using namespace logging::trivial;
    src::severity_logger< severity_level > lg;
    BOOST_LOG_SEV(lg, info) <<" Initialization";
    return 0;
}

#else
class TestClass
{
public:
  static int testFunc()
  {
      LOG("This is a testFunc.");
      return 0;
  }
};

int main()
{
    LOG("Going to call testFunc.");
    timer.enter("TestFunc");
    for(int i=0,iend=10000;i<iend;i++)
        TestClass::testFunc();
    timer.leave("TestFunc");
    return 0;
}

#endif
