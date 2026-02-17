"use client";

import { useState } from "react";
import { FilterOptions } from "@/lib/api";

interface FilterPanelProps {
  options: FilterOptions;
  selected: {
    year: number[];
    region: string[];
    industry: string[];
    article: string[];
  };
  onChange: (filterType: string, values: any[]) => void;
}

// Маппинг субъектов РФ к УФАС для фильтрации
const REGION_TO_UFAS: Record<string, string> = {
  "Республика Адыгея": "Адыгейское УФАС",
  "Республика Алтай": "Алтайское УФАС",
  "Амурская область": "Амурское УФАС",
  "Архангельская область": "Архангельское УФАС",
  "Астраханская область": "Астраханское УФАС",
  "Республика Башкортостан": "Башкортостанское УФАС",
  "Белгородская область": "Белгородское УФАС",
  "Брянская область": "Брянское УФАС",
  "Республика Бурятия": "Бурятское УФАС",
  "Волгоградская область": "Волгоградское УФАС",
  "Вологодская область": "Вологодское УФАС",
  "Воронежская область": "Воронежское УФАС",
  "Республика Дагестан": "Дагестанское УФАС",
  "Республика Калмыкия": "Калмыцкое УФАС",
  "Республика Крым": "Крымское МУФАС",
  "Краснодарский край": "Краснодарское УФАС",
  "Красноярский край": "Красноярское УФАС",
  "Курганская область": "Курганское УФАС",
  "Курская область": "Курское УФАС",
  "Ленинградская область": "Ленинградское УФАС",
  "Липецкая область": "Липецкое УФАС",
  "Ивановская область": "Ивановское УФАС",
  "Костромская область": "Костромское УФАС",
  "Московская область": "Московское областное УФАС",
  "Орловская область": "Орловское УФАС",
  "Рязанская область": "Рязанское УФАС",
  "Смоленская область": "Смоленское УФАС",
  "Тамбовская область": "Тамбовское УФАС",
  "Тверская область": "Тверское УФАС",
  "Тульская область": "Тульское УФАС",
  "Ярославская область": "Ярославское УФАС",
  "Москва": "Московское УФАС",
  "Санкт-Петербург": "Санкт-Петербургское УФАС",
  "Севастополь": "Севастопольское УФАС",
  "Город федерального значения Москва": "Московское УФАС",
  "Город федерального значения Санкт-Петербург": "Санкт-Петербургское УФАС",
  "Город федерального значения Севастополь": "Севастопольское УФАС",
  "Республика Карелия": "Карельское УФАС",
  "Республика Коми": "Коми УФАС",
  "Калининградская область": "Калининградское УФАС",
  "Мурманская область": "Мурманское УФАС",
  "Новгородская область": "Новгородское УФАС",
  "Псковская область": "Псковское УФАС",
  "Ненецкий автономный округ": "Ненецкое УФАС",
  "Республика Саха (Якутия)": "Якутское УФАС",
  "Республика Саха": "Якутское УФАС",
  "Камчатский край": "Камчатское УФАС",
  "Приморский край": "Приморское УФАС",
  "Хабаровский край": "Хабаровское УФАС",
  "Магаданская область": "Магаданское УФАС",
  "Сахалинская область": "Сахалинское УФАС",
  "Еврейская автономная область": "Еврейское УФАС",
  "Чукотский автономный округ": "Чукотское УФАС",
  "Республика Тыва": "Тывинское УФАС",
  "Республика Хакасия": "Хакасское УФАС",
  "Забайкальский край": "Забайкальское УФАС",
  "Иркутская область": "Иркутское УФАС",
  "Кемеровская область": "Кемеровское УФАС",
  "Новосибирская область": "Новосибирское УФАС",
  "Омская область": "Омское УФАС",
  "Томская область": "Томское УФАС",
  "Свердловская область": "Свердловское УФАС",
  "Тюменская область": "Тюменское УФАС",
  "Челябинская область": "Челябинское УФАС",
  "Ханты-Мансийский автономный округ — Югра": "Ханты-Мансийское УФАС",
  "Ямало-Ненецкий автономный округ": "Ямало-Ненецкое УФАС",
  "Республика Марий Эл": "Марийское УФАС",
  "Республика Мордовия": "Мордовское УФАС",
  "Республика Татарстан": "Татарстанское УФАС",
  "Удмуртская Республика": "Удмуртское УФАС",
  "Чувашская Республика": "Чувашское УФАС",
  "Кировская область": "Кировское УФАС",
  "Нижегородская область": "Нижегородское УФАС",
  "Оренбургская область": "Оренбургское УФАС",
  "Пензенская область": "Пензенское УФАС",
  "Ульяновская область": "Ульяновское УФАС",
  "Самарская область": "Самарское УФАС",
  "Саратовская область": "Саратовское УФАС",
  "Пермский край": "Пермское УФАС",
  "Республика Ингушетия": "Ингушское УФАС",
  "Кабардино-Балкарская Республика": "Кабардино-Балкарское УФАС",
  "Карачаево-Черкесская Республика": "Карачаево-Черкесское УФАС",
  "Республика Северная Осетия — Алания": "Северо-Осетинское УФАС",
  "Чеченская Республика": "Чеченское УФАС",
  "Ростовская область": "Ростовское УФАС",
  "Ставропольский край": "Ставропольское УФАС",
};

// Функция для отображения краткого названия региона
const displayRegionName = (region: string): string => {
  const shortNames: Record<string, string> = {
    "Город федерального значения Москва": "Москва",
    "Город федерального значения Санкт-Петербург": "Санкт-Петербург",
    "Город федерального значения Севастополь": "Севастополь",
  };
  return shortNames[region] || region;
};

// Маппинг статей для фильтрации
const ARTICLE_TO_LEGAL: Record<string, string[]> = {
  "ст. 2": ["ст. 2"],
  "ст. 3": ["ст. 3"],
  "ст. 5": ["ст. 5", "ч. 1 ст. 5", "ч. 2 ст. 5", "ч. 3 ст. 5", "ч. 4 ст. 5", "ч. 5 ст. 5", "ч. 6 ст. 5", "ч. 7 ст. 5", "ч. 9 ст. 5", "ч. 10 ст. 5", "ч. 11 ст. 5"],
  "ст. 6": ["ст. 6"],
  "ст. 7": ["ст. 7"],
  "ст. 8": ["ст. 8"],
  "ст. 9": ["ст. 9"],
  "ст. 12": ["ст. 12"],
  "ст. 14": ["ст. 14"],
  "ст. 16": ["ст. 16"],
  "ст. 19": ["ст. 19"],
  "ст. 21": ["ст. 21"],
  "ст. 25": ["ст. 25"],
  "ст. 27": ["ст. 27"],
  "ст. 28": ["ст. 28"],
};

export function FilterPanel({ options, selected, onChange }: FilterPanelProps) {
  const [openSection, setOpenSection] = useState<string | null>("year");
  const [expandedIndustry, setExpandedIndustry] = useState<string | null>(null);
  const [expandedSubIndustry, setExpandedSubIndustry] = useState<string | null>(null);
  const [expandedRegion, setExpandedRegion] = useState<string | null>(null);
  const [expandedArticle, setExpandedArticle] = useState<string | null>(null);

  const toggleSection = (section: string) => {
    setOpenSection(openSection === section ? null : section);
  };

  const handleCheckboxChange = (section: string, value: string | number, checked: boolean) => {
    const currentValues = selected[section as keyof typeof selected];
    let newValues: any[];
    
    if (checked) {
      newValues = [...currentValues, value];
    } else {
      newValues = currentValues.filter(v => v !== value);
    }
    
    onChange(section, newValues);
  };

  // ============ РЕГИОНЫ ============
  
  // Выбор всех УФАС федерального округа при клике на чекбокс ФО
  const handleRegionGroupCheckbox = (groupName: string, regions: string[]) => {
    const ufasList: string[] = [];
    for (const r of regions) {
      const ufasName = REGION_TO_UFAS[r];
      if (ufasName && options.regions.includes(ufasName)) {
        ufasList.push(ufasName);
      }
    }
    
    // Проверяем, сколько УФАС из этого ФО уже выбрано
    const selectedFromGroup = selected.region.filter(r => ufasList.includes(r));
    
    if (selectedFromGroup.length > 0) {
      // Если уже выбрано - убираем все УФАС этого ФО
      const newValues = selected.region.filter(r => !ufasList.includes(r));
      onChange("region", newValues);
    } else {
      // Если не выбрано - добавляем все УФАС
      onChange("region", [...selected.region, ...ufasList]);
    }
  };

  // Проверка выбора - ФО выбран если выбраны все или несколько УФАС
  const isRegionGroupSelected = (groupName: string, regions: string[]) => {
    const ufasList: string[] = [];
    for (const r of regions) {
      const ufasName = REGION_TO_UFAS[r];
      if (ufasName && options.regions.includes(ufasName)) {
        ufasList.push(ufasName);
      }
    }
    return ufasList.some(ufas => selected.region.includes(ufas));
  };

  // ============ СТАТЬИ ============
  
  // Выбор всех частей статьи при клике на чекбокс статьи
  const handleArticleGroupCheckbox = (articleName: string, parts: {name: string, count: number}[]) => {
    const articleList: string[] = [articleName];
    for (const p of parts) {
      articleList.push(`${p.name} ст. ${articleName.replace('ст. ', '')}`);
    }
    
    // Проверяем, сколько частей уже выбрано
    const selectedFromArticle = selected.article.filter(a => articleList.includes(a));
    
    if (selectedFromArticle.length > 0) {
      // Убираем все части этой статьи
      const newValues = selected.article.filter(a => !articleList.includes(a));
      onChange("article", newValues);
    } else {
      // Добавляем все части
      onChange("article", [...selected.article, ...articleList]);
    }
  };

  // Проверка выбора - статья выбрана если выбраны все или несколько частей
  const isArticleGroupSelected = (articleName: string, parts: {name: string, count: number}[]) => {
    const articleList: string[] = [articleName];
    for (const p of parts) {
      articleList.push(`${p.name} ст. ${articleName.replace('ст. ', '')}`);
    }
    return articleList.some(a => selected.article.includes(a));
  };

  // ============ ОТРАСЛИ ============
  
  const handleIndustryGroupClick = (groupName: string, subIndustries: {name: string, sub_industries?: {name: string, count: number}[]}[]) => {
    const groupIndustries = selected.industry.filter(ind => ind === groupName || ind.startsWith(groupName + " / "));
    
    if (groupIndustries.length > 0) {
      const toRemove = selected.industry.filter(ind => ind === groupName || ind.startsWith(groupName + " / "));
      const newValues = selected.industry.filter(ind => !toRemove.includes(ind));
      onChange("industry", newValues);
    } else {
      const toAdd = [groupName];
      for (const sub of subIndustries) {
        toAdd.push(`${groupName} / ${sub.name}`);
        if (sub.sub_industries) {
          for (const subsub of sub.sub_industries) {
            toAdd.push(`${groupName} / ${sub.name} / ${subsub.name}`);
          }
        }
      }
      onChange("industry", [...selected.industry, ...toAdd]);
    }
    
    setExpandedIndustry(expandedIndustry === groupName ? null : groupName);
    setExpandedSubIndustry(null);
  };

  const handleSubIndustryClick = (groupName: string, subName: string, subSubIndustries?: {name: string, count: number}[]) => {
    const subFullName = `${groupName} / ${subName}`;
    const currentSubIndustries = selected.industry.filter(ind => ind === subFullName || ind.startsWith(subFullName + " / "));
    
    if (currentSubIndustries.length > 0) {
      const toRemove = selected.industry.filter(ind => ind === subFullName || ind.startsWith(subFullName + " / "));
      const newValues = selected.industry.filter(ind => !toRemove.includes(ind));
      onChange("industry", newValues);
    } else {
      const toAdd = [subFullName];
      if (subSubIndustries) {
        for (const subsub of subSubIndustries) {
          toAdd.push(`${subFullName} / ${subsub.name}`);
        }
      }
      onChange("industry", [...selected.industry, ...toAdd]);
    }
    
    setExpandedSubIndustry(expandedSubIndustry === subFullName ? null : subFullName);
  };

  const isIndustryGroupSelected = (groupName: string) => {
    return selected.industry.some(ind => ind === groupName || ind.startsWith(groupName + " / "));
  };

  const isSubIndustrySelected = (groupName: string, subName: string) => {
    const subFullName = `${groupName} / ${subName}`;
    return selected.industry.some(ind => ind === subFullName || ind.startsWith(subFullName + " / "));
  };

  return (
    <div className="space-y-4">
      {/* Год решения */}
      <div className="border rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection("year")}
          className="w-full flex items-center justify-between p-4 bg-muted/50 hover:bg-muted transition-colors"
        >
          <span className="font-medium">Год решения ФАС</span>
          <div className="flex items-center gap-2">
            {selected.year.length > 0 && (
              <span className="bg-primary text-primary-foreground text-xs px-2 py-0.5 rounded-full">
                {selected.year.length}
              </span>
            )}
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={`transition-transform ${openSection === "year" ? "rotate-180" : ""}`}>
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </button>
        {openSection === "year" && (
          <div className="p-4 border-t max-h-[200px] overflow-y-auto">
            {options.years.length > 0 ? (
              <div className="space-y-2">
                {options.years.map((year) => (
                  <label key={year} className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selected.year.includes(year)}
                      onChange={(e) => handleCheckboxChange("year", year, e.target.checked)}
                      className="w-4 h-4 rounded border-border text-primary focus:ring-primary"
                    />
                    <span className="text-sm">{year}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Нет данных</p>
            )}
          </div>
        )}
      </div>

      {/* Регион - иерархия по федеральным округам */}
      <div className="border rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection("region")}
          className="w-full flex items-center justify-between p-4 bg-muted/50 hover:bg-muted transition-colors"
        >
          <span className="font-medium">Регион</span>
          <div className="flex items-center gap-2">
            {selected.region.length > 0 && (
              <span className="bg-primary text-primary-foreground text-xs px-2 py-0.5 rounded-full">
                {selected.region.length}
              </span>
            )}
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={`transition-transform ${openSection === "region" ? "rotate-180" : ""}`}>
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </button>
        
        {openSection === "region" && (
          <div className="p-4 border-t max-h-[350px] overflow-y-auto">
            {options.region_groups && options.region_groups.length > 0 ? (
              <div className="space-y-1">
                {options.region_groups.map((group) => {
                  const isSelected = isRegionGroupSelected(group.name, group.regions);
                  const hasRegions = group.regions && group.regions.length > 0;
                  
                  return (
                    <div key={group.name}>
                      {/* Федеральный округ */}
                      <div className="flex items-center gap-2 p-2 rounded">
                        {/* Чекбокс - выбирает все регионы */}
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => hasRegions ? handleRegionGroupCheckbox(group.name, group.regions) : null}
                          className="w-4 h-4 rounded border-border text-primary"
                          disabled={!hasRegions}
                        />
                        <span className="text-sm flex-1 font-medium">{group.name}</span>
                        <span className="text-xs text-muted-foreground">({group.count})</span>
                        {/* Стрелка - только раскрывает, не выбирает */}
                        {hasRegions && (
                          <button
                            onClick={() => setExpandedRegion(expandedRegion === group.name ? null : group.name)}
                            className="p-1 hover:bg-muted rounded"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={`transition-transform ${expandedRegion === group.name ? "rotate-90" : ""}`}>
                              <polyline points="9 18 15 12 9 6" />
                            </svg>
                          </button>
                        )}
                      </div>
                      
                      {/* Регионы */}
                      {hasRegions && expandedRegion === group.name && (
                        <div className="ml-6 space-y-1">
                          {group.regions?.map((region) => {
                            const ufasName = REGION_TO_UFAS[region];
                            const isRegionSelected = ufasName ? selected.region.includes(ufasName) : false;
                            
                            return (
                              <label key={region} className="flex items-center gap-2 p-1 rounded cursor-pointer hover:bg-muted">
                                <input
                                  type="checkbox"
                                  checked={isRegionSelected}
                                  onChange={(e) => ufasName ? handleCheckboxChange("region", ufasName, e.target.checked) : null}
                                  className="w-3.5 h-3.5 rounded border-border text-primary"
                                />
                                <span className="text-sm">{displayRegionName(region)}</span>
                              </label>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : options.regions.length > 0 ? (
              <div className="space-y-2">
                {options.regions.slice(0, 50).map((region) => (
                  <label key={region} className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selected.region.includes(region)}
                      onChange={(e) => handleCheckboxChange("region", region, e.target.checked)}
                      className="w-4 h-4 rounded border-border text-primary focus:ring-primary"
                    />
                    <span className="text-sm truncate">{displayRegionName(region)}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Нет данных</p>
            )}
          </div>
        )}
      </div>

      {/* Отрасль - 3 уровня */}
      <div className="border rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection("industry")}
          className="w-full flex items-center justify-between p-4 bg-muted/50 hover:bg-muted transition-colors"
        >
          <span className="font-medium">Отрасль</span>
          <div className="flex items-center gap-2">
            {selected.industry.length > 0 && (
              <span className="bg-primary text-primary-foreground text-xs px-2 py-0.5 rounded-full">
                {selected.industry.length}
              </span>
            )}
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={`transition-transform ${openSection === "industry" ? "rotate-180" : ""}`}>
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </button>
        
        {openSection === "industry" && (
          <div className="p-4 border-t max-h-[350px] overflow-y-auto">
            {options.industry_groups && options.industry_groups.length > 0 ? (
              <div className="space-y-1">
                {options.industry_groups.map((group) => {
                  const isSelected = isIndustryGroupSelected(group.name);
                  const hasSub = group.sub_industries && group.sub_industries.length > 0;
                  
                  return (
                    <div key={group.name}>
                      <div className="flex items-center gap-2 p-2 rounded">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => hasSub ? handleIndustryGroupClick(group.name, group.sub_industries || []) : handleCheckboxChange("industry", group.name, !isSelected)}
                          className="w-4 h-4 rounded border-border text-primary"
                          onClick={(e) => e.stopPropagation()}
                        />
                        <span className="text-sm flex-1">{group.name}</span>
                        <span className="text-xs text-muted-foreground">({group.count})</span>
                        {hasSub && (
                          <button
                            onClick={() => setExpandedIndustry(expandedIndustry === group.name ? null : group.name)}
                            className="p-1 hover:bg-muted rounded"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={`transition-transform ${expandedIndustry === group.name ? "rotate-90" : ""}`}>
                              <polyline points="9 18 15 12 9 6" />
                            </svg>
                          </button>
                        )}
                      </div>
                      
                      {hasSub && expandedIndustry === group.name && (
                        <div className="ml-4 space-y-1">
                          {group.sub_industries?.map((sub) => {
                            const isSubSelected = isSubIndustrySelected(group.name, sub.name);
                            const hasSubSub = sub.sub_industries && sub.sub_industries.length > 0;
                            
                            return (
                              <div key={sub.name}>
                                <div className="flex items-center gap-2 p-1.5 rounded">
                                  <input
                                    type="checkbox"
                                    checked={isSubSelected}
                                    onChange={() => hasSubSub ? handleSubIndustryClick(group.name, sub.name, sub.sub_industries) : handleCheckboxChange("industry", `${group.name} / ${sub.name}`, !isSubSelected)}
                                    className="w-3.5 h-3.5 rounded border-border text-primary"
                                    onClick={(e) => e.stopPropagation()}
                                  />
                                  <span className="text-sm flex-1">{sub.name}</span>
                                  <span className="text-xs text-muted-foreground">({sub.count})</span>
                                  {hasSubSub && (
                                    <button
                                      onClick={() => setExpandedSubIndustry(expandedSubIndustry === `${group.name} / ${sub.name}` ? null : `${group.name} / ${sub.name}`)}
                                      className="p-1 hover:bg-muted rounded"
                                    >
                                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={`transition-transform ${expandedSubIndustry === `${group.name} / ${sub.name}` ? "rotate-90" : ""}`}>
                                        <polyline points="9 18 15 12 9 6" />
                                      </svg>
                                    </button>
                                  )}
                                </div>
                                
                                {hasSubSub && expandedSubIndustry === `${group.name} / ${sub.name}` && (
                                  <div className="ml-4 space-y-1">
                                    {sub.sub_industries?.map((subsub) => {
                                      const fullName = `${group.name} / ${sub.name} / ${subsub.name}`;
                                      const isSubSubSelected = selected.industry.includes(fullName);
                                      
                                      return (
                                        <label key={subsub.name} className="flex items-center gap-2 p-1 rounded cursor-pointer hover:bg-muted">
                                          <input
                                            type="checkbox"
                                            checked={isSubSubSelected}
                                            onChange={() => handleCheckboxChange("industry", fullName, !isSubSubSelected)}
                                            className="w-3 h-3 rounded border-border text-primary"
                                          />
                                          <span className="text-sm">{subsub.name}</span>
                                          <span className="text-xs text-muted-foreground">({subsub.count})</span>
                                        </label>
                                      );
                                    })}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Нет данных</p>
            )}
          </div>
        )}
      </div>

      {/* Статья закона - иерархия */}
      <div className="border rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection("article")}
          className="w-full flex items-center justify-between p-4 bg-muted/50 hover:bg-muted transition-colors"
        >
          <span className="font-medium">Статья закона</span>
          <div className="flex items-center gap-2">
            {selected.article.length > 0 && (
              <span className="bg-primary text-primary-foreground text-xs px-2 py-0.5 rounded-full">
                {selected.article.length}
              </span>
            )}
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={`transition-transform ${openSection === "article" ? "rotate-180" : ""}`}>
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </button>
        
        {openSection === "article" && (
          <div className="p-4 border-t max-h-[350px] overflow-y-auto">
            {options.article_groups && options.article_groups.length > 0 ? (
              <div className="space-y-1">
                {options.article_groups.map((group) => {
                  const isSelected = isArticleGroupSelected(group.name, group.parts);
                  const hasParts = group.parts && group.parts.length > 0;
                  
                  return (
                    <div key={group.name}>
                      {/* Статья */}
                      <div className="flex items-center gap-2 p-2 rounded">
                        {/* Чекбокс - выбирает все части */}
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => handleArticleGroupCheckbox(group.name, group.parts)}
                          className="w-4 h-4 rounded border-border text-primary"
                        />
                        <span className="text-sm flex-1 font-medium">{group.name}</span>
                        <span className="text-xs text-muted-foreground">({group.count})</span>
                        {/* Стрелка - только раскрывает, не выбирает */}
                        {hasParts && (
                          <button
                            onClick={() => setExpandedArticle(expandedArticle === group.name ? null : group.name)}
                            className="p-1 hover:bg-muted rounded"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={`transition-transform ${expandedArticle === group.name ? "rotate-90" : ""}`}>
                              <polyline points="9 18 15 12 9 6" />
                            </svg>
                          </button>
                        )}
                      </div>
                      
                      {/* Части статьи */}
                      {hasParts && expandedArticle === group.name && (
                        <div className="ml-6 space-y-1">
                          {group.parts?.map((part) => {
                            const fullName = `${part.name} ст. ${group.name.replace('ст. ', '')}`;
                            const isPartSelected = selected.article.includes(fullName);
                            
                            return (
                              <label key={part.name} className="flex items-center gap-2 p-1 rounded cursor-pointer hover:bg-muted">
                                <input
                                  type="checkbox"
                                  checked={isPartSelected}
                                  onChange={(e) => handleCheckboxChange("article", fullName, e.target.checked)}
                                  className="w-3.5 h-3.5 rounded border-border text-primary"
                                />
                                <span className="text-sm">{part.name}</span>
                              </label>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : options.articles.length > 0 ? (
              <div className="space-y-2">
                {options.articles.slice(0, 30).map((article) => (
                  <label key={article} className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selected.article.includes(article)}
                      onChange={(e) => handleCheckboxChange("article", article, e.target.checked)}
                      className="w-4 h-4 rounded border-border text-primary focus:ring-primary"
                    />
                    <span className="text-sm">{article}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Нет данных</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
