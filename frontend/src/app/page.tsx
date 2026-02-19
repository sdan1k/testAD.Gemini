"use client";

import { useState, useEffect, useRef } from "react";
import { SearchForm } from "@/components/search-form";
import { CaseCard } from "@/components/case-card";
import { FilterPanel } from "@/components/filter-panel";
import { HelpButton } from "@/components/help-button";
import { searchCases, checkHealth, getFilterOptions, CaseResult, FilterOptions } from "@/lib/api";

export default function Home() {
  const [results, setResults] = useState<CaseResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<string | null>(null);
  const [totalCases, setTotalCases] = useState<number>(0);
  const [serverStatus, setServerStatus] = useState<"checking" | "online" | "offline">("checking");
  const [showFilters, setShowFilters] = useState(false);
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);
  const [selectedFilters, setSelectedFilters] = useState({
    year: [] as number[],
    region: [] as string[],
    industry: [] as string[],
    article: [] as string[],
  });
  const [message, setMessage] = useState<string | null>(null);
  const [searchPerformed, setSearchPerformed] = useState(false);
  const [cursorPos, setCursorPos] = useState({ x: -100, y: -100 });
  const [isHovering, setIsHovering] = useState(false);
  
  const [isClient, setIsClient] = useState(false);

  // Генерируем плавающие круги только на клиенте
  useEffect(() => {
    setIsClient(true);
    
    // Создаем круги по коду пользователя
    const container = document.getElementById('floatingCircles');
    if (!container) return;
    
    const circleCount = 12;
    for (let i = 0; i < circleCount; i++) {
      const circle = document.createElement('div');
      circle.className = 'floating-circle';
      
      const size = Math.random() * 120 + 40;
      const posX = Math.random() * 100;
      const posY = Math.random() * 100;
      const duration = Math.random() * 25 + 20;
      const delay = Math.random() * 10;
      
      circle.style.width = `${size}px`;
      circle.style.height = `${size}px`;
      circle.style.left = `${posX}%`;
      circle.style.top = `${posY}%`;
      
      const animationName = `floatCircle${i}`;
      circle.style.animation = `${animationName} ${duration}s infinite ${delay}s ease-in-out`;
      
      container.appendChild(circle);
      
      const style = document.createElement('style');
      const dirX1 = Math.random() > 0.5 ? 1 : -1;
      const dirY1 = Math.random() > 0.5 ? 1 : -1;
      const moveX1 = Math.random() * 150 - 75;
      const moveY1 = Math.random() * 150 - 75;
      const moveX2 = Math.random() * 200 - 100;
      const moveY2 = Math.random() * 200 - 100;
      const moveX3 = Math.random() * 150 - 75;
      const moveY3 = Math.random() * 150 - 75;
      
      style.textContent = `
        @keyframes ${animationName} {
            0%, 100% { transform: translate(0, 0); }
            25% { transform: translate(${dirX1 * moveX1}px, ${dirY1 * moveY1}px); }
            50% { transform: translate(${dirX1 * moveX2}px, ${dirY1 * moveY2}px); }
            75% { transform: translate(${dirX1 * moveX3}px, ${dirY1 * moveY3}px); }
        }
      `;
      document.head.appendChild(style);
    }
  }, []);

  // Отслеживание курсора
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setCursorPos({ x: e.clientX, y: e.clientY });
    };
    
    const handleMouseEnter = () => setIsHovering(true);
    const handleMouseLeave = () => setIsHovering(false);
    
    document.addEventListener('mousemove', handleMouseMove);
    
    const interactiveElements = document.querySelectorAll('button, input, a, [role="button"]');
    interactiveElements.forEach(el => {
      el.addEventListener('mouseenter', handleMouseEnter);
      el.addEventListener('mouseleave', handleMouseLeave);
    });
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      interactiveElements.forEach(el => {
        el.removeEventListener('mouseenter', handleMouseEnter);
        el.removeEventListener('mouseleave', handleMouseLeave);
      });
    };
  }, []);

  // Проверка состояния сервера
  useEffect(() => {
    const checkServer = async () => {
      try {
        const health = await checkHealth();
        if (health.status === "ok" && health.data_loaded) {
          setServerStatus("online");
          setTotalCases(health.total_cases);
          try {
            const options = await getFilterOptions();
            setFilterOptions(options);
          } catch (e) {
            console.error("Failed to load filter options:", e);
          }
        } else {
          setServerStatus("offline");
        }
      } catch {
        setServerStatus("offline");
      }
    };
    checkServer();
  }, []);

  const handleSearch = async (query: string) => {
    setIsLoading(true);
    setError(null);
    setLastQuery(query);
    setMessage(null);
    setSearchPerformed(true);

    try {
      const filters = {
        year: selectedFilters.year.length > 0 ? selectedFilters.year : undefined,
        region: selectedFilters.region.length > 0 ? selectedFilters.region : undefined,
        industry: selectedFilters.industry.length > 0 ? selectedFilters.industry : undefined,
        article: selectedFilters.article.length > 0 ? selectedFilters.article : undefined,
      };
      
      const response = await searchCases(query, 20, filters);
      setResults(response.results);
      setTotalCases(response.total_cases);
      if (response.message) {
        setMessage(response.message);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Произошла ошибка при поиске");
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFilterChange = async (filterType: string, values: any[]) => {
    // Сначала создаём новые фильтры
    const newFilters = {
      ...selectedFilters,
      [filterType]: values,
    };
    
    // Обновляем состояние
    setSelectedFilters(newFilters);
    
    // Если уже был выполнен поиск, повторяем его с новыми фильтрами
    if (lastQuery && searchPerformed) {
      setIsLoading(true);
      try {
        const filterParams = {
          year: newFilters.year.length > 0 ? newFilters.year : undefined,
          region: newFilters.region.length > 0 ? newFilters.region : undefined,
          industry: newFilters.industry.length > 0 ? newFilters.industry : undefined,
          article: newFilters.article.length > 0 ? newFilters.article : undefined,
        };
        
        const response = await searchCases(lastQuery, 20, filterParams);
        setResults(response.results);
        setTotalCases(response.total_cases);
      } catch (err) {
        console.error("Error updating search with filters:", err);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleClearFilters = async () => {
    const emptyFilters = {
      year: [],
      region: [],
      industry: [],
      article: [],
    };
    
    setSelectedFilters(emptyFilters);
    
    // Если уже был выполнен поиск, повторяем его без фильтров
    if (lastQuery && searchPerformed) {
      setIsLoading(true);
      try {
        const response = await searchCases(lastQuery, 20, undefined);
        setResults(response.results);
        setTotalCases(response.total_cases);
      } catch (err) {
        console.error("Error clearing filters:", err);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Плавающие фиолетовые круги */}
      <div className="floating-circles-bg" id="floatingCircles"></div>
      
      {/* Кастомный курсор */}
      <div 
        className={`cursor-glow ${isHovering ? 'hidden' : ''}`}
        style={{ left: cursorPos.x, top: cursorPos.y }} 
      />
      
      {/* Header - адаптивный */}
      <header className="header flex-col md:flex-row h-auto md:h-16 py-2 md:py-0">
        <div className="flex-1">
          <h1 className="text-lg md:text-2xl font-bold">Поиск практики ФАС</h1>
          <p className="text-muted-foreground text-xs md:text-sm mt-0 md:mt-1 hidden md:block">
            Умный поиск по 7000+ решений ФАС
          </p>
        </div>
        <div className="flex items-center gap-2 md:gap-4 mt-2 md:mt-0">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`btn-secondary flex items-center gap-2 ${showFilters ? 'bg-primary/10' : ''}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
            </svg>
            Фильтры
          </button>
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                serverStatus === "online" ? "bg-green-500" : serverStatus === "offline" ? "bg-red-500" : "bg-yellow-500 animate-pulse"
              }`}
            />
            <span className="text-sm text-muted-foreground">
              {serverStatus === "online" ? `${totalCases.toLocaleString("ru-RU")} дел` : serverStatus === "offline" ? "Сервер недоступен" : "Проверка..."}
            </span>
          </div>
        </div>
      </header>

      <div className="flex">
        {showFilters && filterOptions && (
          <>
            {/* Overlay для мобильных - закрывает при клике */}
            <div className="lg:hidden fixed inset-0 z-30 bg-black/50" onClick={() => setShowFilters(false)} />
            <aside className="w-[320px] border-r bg-card h-screen overflow-hidden flex flex-col fixed left-4 top-0 pt-16 z-40 rounded-l-lg md:left-0 md:w-full md:max-w-[320px] md:rounded-none lg:left-4 lg:w-[320px] lg:rounded-l-lg">
              <div className="flex items-center justify-between p-4 border-b bg-card shrink-0">
                <h2 className="text-lg font-semibold">Фильтры</h2>
                <button onClick={handleClearFilters} className="text-sm px-3 py-1.5 rounded-md bg-primary/10 text-primary hover:bg-primary/20 transition-colors font-medium">
                  Очистить все
                </button>
              </div>
              <div className="p-4 flex-1 overflow-y-auto">
                <FilterPanel options={filterOptions} selected={selectedFilters} onChange={handleFilterChange} />
              </div>
            </aside>
          </>
        )}

        <main className={`flex-1 container mx-auto px-4 py-8 pt-20 transition-all duration-300 ${showFilters ? 'xl:ml-[336px]' : ''}`}>
          <div className="max-w-3xl mx-auto mb-8">
            <SearchForm onSearch={handleSearch} isLoading={isLoading} />
          </div>

          {error && (
            <div className="max-w-3xl mx-auto mb-8">
              <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
                <p className="text-destructive text-sm">{error}</p>
              </div>
            </div>
          )}

          {message && (
            <div className="max-w-3xl mx-auto mb-8">
              <div className="bg-primary/10 border border-primary/20 rounded-lg p-4">
                <p className="text-primary text-sm">{message}</p>
              </div>
            </div>
          )}

          {results.length > 0 && (
            <div className="max-w-3xl mx-auto">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-medium">
                  Результаты поиска
                  {lastQuery && <span className="text-muted-foreground font-normal ml-2">по запросу «{lastQuery}»</span>}
                </h2>
                <span className="text-sm text-muted-foreground">Найдено: {results.length}</span>
              </div>
              <div className="space-y-4">
                {results.map((caseData, index) => (
                  <CaseCard key={caseData.docId || index} caseData={caseData} rank={index + 1} />
                ))}
              </div>
            </div>
          )}

          {isLoading && (
            <div className="max-w-3xl mx-auto">
              <div className="empty-state">
                <div className="spinner" style={{ width: 48, height: 48 }} />
                <p className="text-lg font-medium mt-4">Поиск по базе решений...</p>
                <p className="text-muted-foreground mt-2">Анализируем контекст и юридические формулировки</p>
              </div>
            </div>
          )}

          {!isLoading && !searchPerformed && results.length === 0 && !error && (
            <div className="max-w-3xl mx-auto text-center py-12">
              <div className="text-6xl mb-4">🔍</div>
              <h2 className="text-xl font-medium mb-2">Введите запрос для поиска</h2>
              <p className="text-muted-foreground max-w-md mx-auto">
                Система найдет похожие решения ФАС по смыслу вашего запроса. Опишите ситуацию или вставьте текст рекламы.
              </p>
            </div>
          )}

          {!isLoading && searchPerformed && results.length === 0 && error === null && (
            <div className="max-w-3xl mx-auto">
              <div className="empty-state">
                <div className="text-6xl mb-4">📋</div>
                <h2 className="text-xl font-medium mb-2">Мы не нашли точных совпадений</h2>
                <p className="text-muted-foreground max-w-md mx-auto">
                  Похоже, таких формулировок в нашей базе еще не встречалось.
                  <br /><br />
                  Попробуйте переформулировать запрос, сбросить фильтры или описать суть.
                </p>
              </div>
            </div>
          )}
        </main>
      </div>

      <footer className="border-t mt-auto">
        <div className="container mx-auto px-4 py-4">
          <p className="text-center text-sm text-muted-foreground">
            Семантический поиск на основе эмбеддингов Google Gemini • Данные из открытых решений ФАС
          </p>
        </div>
      </footer>

      <HelpButton />
    </div>
  );
}
