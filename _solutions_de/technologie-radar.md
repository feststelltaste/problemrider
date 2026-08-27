---
title: Technologie-Radar
description: Pflege einer expliziten, regelmäßig überprüften Liste, welche
  Technologien die Organisation übernimmt, toleriert oder abschaltet — und
  Einhaltung der Entscheidungen.
category:
- Architecture
- Management
- Dependencies
problems:
- obsolete-technologies
- technology-lock-in
- premature-technology-introduction
- cv-driven-development
- cargo-culting
- dependency-version-conflicts
- technology-isolation
- technology-stack-fragmentation
- rapid-prototyping-becoming-production
- shared-dependencies
- vendor-dependency
- second-system-effect
- dependency-on-supplier
- inappropriate-skillset
- vendor-relationship-strain
- reimplemented-standard-functionality
layout: solution
lang: de
en_slug: technology-radar
related_solutions:
- slug: boring-technologies
  similarity: 0.8
- slug: architecture-decision-records
  similarity: 0.7
- slug: technical-debt-backlog
  similarity: 0.7
- slug: application-portfolio-inventory
  similarity: 0.7
- slug: architecture-review-board
  similarity: 0.7
- slug: architecture-reviews
  similarity: 0.65
---

## Description

Ein Technologie-Radar ist eine gepflegte, veröffentlichte Klassifikation der Technologien, die eine Organisation nutzt: welche die Standardwahl sind, welche unter bestimmten Umständen erlaubt sind, welche erprobt werden und welche auf dem Weg hinaus sind. Er wird in fester Taktung von einer Gruppe überprüft, die die Autorität hat, Elemente zwischen Kategorien zu verschieben. Sein Wert in einer Legacy-Landschaft ist zweifach. Er schränkt die Proliferation künftig ein, sodass das nächste Jahrzehnt nicht weitere fünf Frameworks hinzufügt, gewählt von wem auch immer gerade das Projekt begann. Und er macht Ausmusterung zu einer expliziten, geplanten Entscheidung statt zu etwas, das passiert, wenn eine Technologie aufhört zu funktionieren — was in der Praxis bedeutet: nie, da eine Komponente, die noch läuft, nie dringend ist, bis sie es plötzlich ist.

## How to Apply ◆

> Die charakteristische Legacy-Technologielandschaft ist nicht eine schlechte Wahl, sondern vierzig vernünftige Entscheidungen, die unabhängig über zwanzig Jahre getroffen wurden, jede noch jemanden erfordernd, der sie kennt.

- **Beginnen Sie mit der Inventarisierung dessen, was tatsächlich im Einsatz ist**, einschließlich der Komponenten, an die niemand seit Jahren gedacht hat. Das Inventar allein findet typischerweise mehrere Technologien, für deren Pflege kein aktueller Mitarbeitender qualifiziert ist, und diese Entdeckung ist oft wertvoller als der Radar selbst.
- Nutzen Sie **eine kleine Anzahl klarer Kategorien** mit unterschiedlichen Bedeutungen. Vier funktioniert gut: Standard für neue Arbeit, akzeptabel unter angegebenen Umständen, unter Bewertung mit einem Entscheidungsdatum, und für Ausmusterung geplant. Mehr Kategorien produzieren Debatte über Klassifikation statt über Technologie.
- **Platzieren Sie jedes Element explizit**, einschließlich derer, über die alle müde sind zu streiten. Eine unklassifizierte Technologie ist eine, die wieder eingeführt wird, von jemandem, der nicht wusste, dass es eine Diskussion gegeben hatte.
- Protokollieren Sie kurz, **warum** jedes Element dort sitzt, wo es sitzt. Die Begründung ist es, was den Radar nützlich macht, wenn jemand eine Platzierung anfechten möchte, und das Anfechten von Platzierungen sollte möglich sein — ein unanfechtbarer Radar wird zu einem Hindernis, das umgangen wird.
- Geben Sie **Ausmusterungs-Einträgen ein Datum und einen Verantwortlichen**, oder sie werden sich nicht bewegen. Eine Technologie, die vier aufeinanderfolgende Reviews lang als auslaufend markiert ist, ist eine Technologie, die nicht ausläuft, und der Radar sollte das sichtbar machen statt es zu verbergen.
- **Überprüfen Sie in fester Taktung**, zweimal jährlich für die meisten Organisationen. Häufigere Überprüfung produziert Unruhe; seltenere lässt den Radar von dem abdriften, was Teams tatsächlich tun, an welchem Punkt er ignoriert wird.
- Beziehen Sie **die Personen, die daran gebunden sein werden**, in die Überprüfung ein. Ein von einer Architekturgruppe produzierter und an Teams ausgegebener Radar wird in Dokumenten befolgt und im Code ignoriert. Einer, bei dem jedes Team eine Stimme bei der Platzierung hat, wird von den Teams selbst durchgesetzt.
- Definieren Sie, was passiert, wenn jemand **etwas nutzen möchte, das nicht auf dem Radar ist**: ein angegebener Prozess, eine zeitlich begrenzte Bewertung und eine Entscheidung. Ohne einen Eintrittspfad wird der Radar zu einem Hindernis, und die Antwort ist, die Technologie still einzuführen.
- **Verbinden Sie ihn mit den Abhängigkeits- und End-of-Support-Daten.** Eine Technologie, deren Herstellersupport in vierzehn Monaten endet, sollte sich automatisch auf dem Radar in Richtung Ausmusterung bewegen, nicht dadurch, dass es jemand bemerkt.

## Tradeoffs ⇄

> Ein Radar reduziert Proliferation und macht Ausmusterung zu einer Entscheidung statt einem Unfall, im Austausch für Governance-Overhead und etwas Verlust an Teamautonomie.

**Vorteile:**

- Technologieproliferation verlangsamt sich, was direkt die Anzahl der Fähigkeiten reduziert, die die Organisation aufrechterhalten muss, und die Anzahl der Komponenten, die unwartbar werden können.
- Ausmusterung wird zu einer geplanten Aktivität mit Verantwortlichen und Terminen, statt zu etwas, das durch eine End-of-Support-Ankündigung oder eine Sicherheitswarnung erzwungen wird.
- Die Einführung neuer Technologie unterliegt einem angegebenen Prozess statt davon, wer das Projekt begonnen hat, was der Mechanismus hinter den meisten CV-getriebenen Entscheidungen ist.
- Wiederkehrende Argumente über dieselben Technologieentscheidungen hören auf, Designdiskussionen zu verbrauchen, weil die Entscheidung und ihre Begründung protokolliert sind.
- Einstellung und Schulung können sich auf einen begrenzten Satz von Technologien konzentrieren, was in einer Organisation, die Systeme in sechs Sprachen pflegt, enorm zählt.

**Kosten und Risiken:**

- Er reduziert die Teamautonomie über technische Entscheidungen, was demotivierend ist und die Entwickler vertreiben kann, die diese Autonomie am meisten schätzen.
- Ein von einer zu weit vom Code entfernten Gruppe gepflegter Radar wird zu einem ignorierten Dokument, und der Aufwand seiner Produktion wird doppelt verschwendet — einmal beim Produzieren, einmal in der parallelen inoffiziellen Realität.
- Übermäßig restriktive Klassifikation treibt Experimentieren in den Untergrund, wo es ohne Review geschieht statt gar nicht.
- Reviews verbrauchen zweimal jährlich Zeit von leitenden technischen Personen und erzeugen Debatten, die politisch werden können.
- Ein Radar kann einen veralteten Standard zementieren. Die Kategorie, die sagt "das nutzen wir", ist die am schwersten zu ändernde, und sie kann eine Organisation über den Punkt hinaus auf einer Technologie halten, an dem die Wahl Sinn ergab.

## How It Could Be

Eine Finanzdienstleistungsorganisation inventarisierte ihre Produktionstechnologie als ersten Schritt zu einem Radar und fand 34 verschiedene Laufzeitplattformen, einschließlich vier Programmiersprachen, für die kein aktueller Mitarbeitender Kompetenz beanspruchte, und zwei Message-Broker, die seit über drei Jahren ohne Herstellersupport waren. Der resultierende Radar war bewusst unambitioniert: zwei Standardsprachen, ein benannter Satz akzeptabler Ausnahmen, und elf Elemente mit Ausmusterungsterminen und Verantwortlichen. Die erste Überprüfung sechs Monate später zeigte drei der elf ausgemustert, zwei mit angegebenen Gründen verlängert und sechs unangetastet — was ein direktes Gespräch über Kapazität auslöste, das zuvor nicht möglich gewesen war, weil "wir mustern die aus" jahrelang eine akzeptable Antwort gewesen war, genau weil niemand mitzählte.

Der Eintrittsprozess erwies sich als wichtiger als die Klassifikationen. Ein Team wollte eine Dokumentendatenbank für ein neues Berichtsfeature einführen. Unter dem vorherigen informellen Regime wäre dies entweder still passiert oder von einem Architekten in einem Meeting abgelehnt worden. Stattdessen trat es als unter Bewertung mit einem dreimonatigen Entscheidungsdatum in den Radar ein, mit einer angegebenen Frage: Reduziert es die Berichtsabfragelast genug, um eine neue betriebliche Fähigkeit zu rechtfertigen. Die Bewertung fand, dass es das nicht tat — eine materialisierte Ansicht in der bestehenden Datenbank erreichte den Großteil des Nutzens —, und das Team kam selbst zu diesem Schluss. Der Beitrag des Radars war nicht die Ablehnung, sondern die Anforderung, dass die Frage explizit gestellt und beantwortet wurde.
