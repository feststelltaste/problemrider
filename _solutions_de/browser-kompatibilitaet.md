---
title: Browser-Kompatibilität
description: Sicherstellung der Browser-Kompatibilität durch Nutzung von Webstandards
  und Progressive Enhancement.
category:
- Code
- Dependencies
problems:
- poor-user-experience-ux-design
- technology-lock-in
- high-client-side-resource-consumption
- inefficient-frontend-code
- user-frustration
- customer-dissatisfaction
layout: solution
lang: de
en_slug: browser-compatibility
related_solutions:
- slug: compatibility-testing
  similarity: 0.75
- slug: cross-platform-frameworks
  similarity: 0.75
- slug: compatibility-certification
  similarity: 0.75
- slug: compatibility-as-error
  similarity: 0.7
- slug: compatibility-measurement
  similarity: 0.7
- slug: a-b-testing
  similarity: 0.7
---

## Description

Browser-Kompatibilität ist die Praxis, Web-Frontends gegen standardisierte HTML-, CSS- und JavaScript-APIs zu bauen statt gegen browserspezifisches Verhalten, unter Nutzung von Progressive Enhancement (Kernfunktionalität funktioniert überall, Erweiterungen schichten sich darauf) und Feature Detection statt User-Agent-Sniffing, um zu entscheiden, was ein gegebener Browser handhaben kann. Der Mechanismus schützt ein Frontend vor den zwei Dingen, die browserspezifischen Code über die Zeit fragil machen: Herstellerspezifische APIs verschwinden, wenn der Browser dieses Herstellers eingestellt wird, und User-Agent-Strings werden zu unzuverlässigen Signalen, während Browser ihr Identifikationsverhalten ändern. Legacy-Webanwendungen sind dem besonders ausgesetzt, weil viele während einer Periode gebaut wurden, in der ein bestimmter Browser, oft Internet Explorer, die Zielumgebung eng genug dominierte, dass Entwickler direkt gegen sein proprietäres Verhalten schrieben — ActiveX-Steuerelemente, herstellerspezifische CSS-Präfixe, Quirks-Mode-Rendering — statt gegen die aufkommenden Webstandards jener Zeit. Wenn dieser dominante Browser schließlich sein End of Life erreicht oder der Nutzeranteil einbricht, wird all dieser browserspezifische Code gleichzeitig und oft unsichtbar für den wachsenden Anteil an Nutzern auf standardkonformen Browsern kaputt, da nichts in der Legacy-Codebasis gebaut wurde, um eine andere Rendering-Engine zu erkennen oder elegant zu degradieren. Browser-Kompatibilität nachzurüsten bedeutet, nach diesen nicht-standardmäßigen Abhängigkeiten zu auditieren, sie durch standardbasierte Äquivalente und Polyfills zu ersetzen, wo nötig, und eine explizite, getestete Support-Matrix für die Zukunft zu etablieren, statt einer impliziten Abhängigkeit von welchem Browser auch immer zum Zeitpunkt des Schreibens Standard war.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Übernehmen Sie Progressive Enhancement: Bauen Sie Kernfunktionalität auf Standard-HTML/CSS auf, schichten Sie dann JavaScript-Erweiterungen darüber
- Ersetzen Sie browserspezifische APIs und Herstellerpräfixe durch standardisierte Web-APIs
- Nutzen Sie Feature Detection (z. B. Modernizr oder native Feature-Prüfungen) statt User-Agent-String-Sniffing des Browsers
- Definieren Sie eine Browser-Support-Matrix und testen Sie dagegen in CI mit automatisierten Cross-Browser-Testwerkzeugen
- Führen Sie Polyfills für kritische Features ein, die in älteren Browsern noch in Ihrer Support-Matrix benötigt werden
- Auditieren Sie Legacy-Frontend-Code auf veraltete oder nicht standardmäßige APIs und erstellen Sie einen Sanierungs-Backlog

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Erreicht eine breitere Nutzerbasis, ohne separate Codepfade pro Browser zu pflegen
- Verringert von Nutzern gemeldete Bugs im Zusammenhang mit browserspezifischen Rendering-Problemen
- Macht das Frontend zukunftssicher, indem auf Standards statt proprietäre Features gesetzt wird

**Kosten und Risiken:**
- Progressive Enhancement könnte die Nutzung neuester Browser-Features einschränken
- Cross-Browser-Testing fügt Zeit und Infrastrukturkosten zur CI-Pipeline hinzu
- Die Unterstützung sehr alter Browser kann die Übernahme moderner Frameworks einschränken
- Polyfills erhöhen die Bundle-Größe und könnten subtile Verhaltensunterschiede einführen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein 2010 gebautes Regierungsportal verließ sich stark auf Internet-Explorer-spezifische ActiveX-Steuerelemente und CSS-Hacks. Nachdem IE sein End of Life erreichte, erlebten über 30 % der Nutzer auf modernen Browsern kaputte Layouts und fehlende Funktionalität. Das Team übernahm eine Progressive-Enhancement-Strategie, ersetzte ActiveX-Komponenten durch Standard-Web-APIs und eliminierte browserspezifisches CSS. Innerhalb von vier Monaten sanken browserbezogene Support-Tickets um 80 %.
